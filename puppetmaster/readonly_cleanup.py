"""Process-local ownership of readonly helpers, including failed teardown."""
import logging
import os
import threading
import time


class _Owner:
    def __init__(self, transport, identity):
        self.transport = transport
        self.identity = identity
        self.permit = None
        self.retired = False
        self.busy = threading.Lock()


class CleanupRegistry:
    """Finalizers retire tokens; only explicit maintenance performs I/O."""
    def __init__(self):
        self.pid = os.getpid()
        # GC can run a slot finalizer while registry bookkeeping allocates or
        # releases objects. The finalizer calls retire() in the same thread.
        self.lock = threading.RLock()
        self.owners = {}

    def register(self, transport, identity):
        token = object()
        with self.lock:
            self.owners[token] = _Owner(transport, identity)
        return token

    def retire(self, token):
        if self.pid == os.getpid():
            with self.lock:
                owner = self.owners.get(token)
                if owner is not None:
                    owner.retired = True

    def hold(self, token, permit):
        if permit is None:
            return
        with self.lock:
            owner = self.owners.get(token)
            if owner is not None:
                if owner.permit is not None and owner.permit is not permit:
                    raise RuntimeError("reader admission ownership conflict")
                owner.permit = permit

    def released(self, token, permit):
        with self.lock:
            owner = self.owners.get(token)
            if owner is not None and owner.permit is permit:
                owner.permit = None

    def close(self, token, deadline=None):
        with self.lock:
            owner = self.owners.get(token)
        if owner is None or self.pid != os.getpid():
            return
        acquired = owner.busy.acquire(timeout=1 if deadline is None else max(0, deadline - time.monotonic()))
        if not acquired:
            raise TimeoutError("reader cleanup already in progress")
        try:
            if deadline is None:
                owner.transport.close()
            else:
                owner.transport.close(deadline=deadline)
            if not owner.transport.closed:
                raise RuntimeError('reader cleanup incomplete')
            if owner.permit is not None:
                owner.permit.release()
                owner.permit = None
            with self.lock:
                self.owners.pop(token, None)
        finally:
            owner.busy.release()

    def _collect(self, predicate, limit):
        with self.lock:
            tokens = [token for token, owner in self.owners.items() if predicate(owner)][:limit]
            for token in tokens:
                self.owners[token] = self.owners.pop(token)
        return tokens

    def _close_tokens(self, tokens, deadline):
        for token in tokens:
            try:
                self.close(token, deadline=deadline)
            except BaseException as exc:
                from puppetmaster.readonly import ReapDeferred

                # A deferred reap is the next sweep's work, not a lost helper.
                level = logging.DEBUG if isinstance(exc, ReapDeferred) else logging.WARNING
                logging.getLogger(__name__).log(level, 'Readonly cleanup retained: %s', exc)

    def maintain(self, limit=8, deadline=None):
        self._close_tokens(self._collect(lambda owner: owner.retired, limit), deadline)

    def reap_idle(self, limit=1, deadline=None):
        """Close helpers whose store was collected and which hold no permit.

        Nothing else reaps them before exit, so a long-lived host that builds
        a store per request kept one idle helper interpreter per store.
        """
        self._close_tokens(self._collect(lambda owner: owner.retired and owner.permit is None, limit),
                           deadline)

    def recover(self, identity, deadline):
        tokens = self._collect(lambda owner: owner.retired and owner.permit is not None
                               and owner.identity == identity, 8)
        self._close_tokens(tokens, deadline)

    def shutdown(self):
        self._close_tokens(self._collect(lambda owner: True, len(self.owners)), None)
