# PyPI Name Reassignment Request — `puppetmaster`

This document is the working draft + outreach template for requesting that
PyPI reassign the project name `puppetmaster` (currently held by an
abandoned 2019 single-release package `puppet-master` by
[QED Software](https://github.com/qedsoftware/puppet-master)) to this
project.

## Why `puppetmaster-ai` is the current pip name

PyPI normalizes project names per [PEP 503](https://peps.python.org/pep-0503/#normalized-names)
by lowercasing and collapsing runs of `-`, `_`, and `.` into single `-`.
Under that rule, `puppetmaster` and `puppet-master` are treated as the
same name. PyPI rejected our v0.7.2 upload with HTTP 400 and the
message:

> The name 'puppetmaster' is too similar to an existing project.

So v0.7.2 ships as `puppetmaster-ai` on PyPI (import name remains
`puppetmaster`; the GitHub repo, CLI, and README branding are
unchanged). This document tracks the work to claim the bare name
back so a future v0.7.x can publish as `puppetmaster` directly.

## Status of the existing `puppet-master` project

| Field | Value |
|---|---|
| PyPI page | https://pypi.org/project/puppet-master/ |
| Author | Quantitative Engineering Design Inc. (qedsoftware) |
| Home page | https://github.com/qedsoftware/puppet-master |
| Total releases | 1 (`0.1.0`) |
| First upload | 2019-08-20 |
| Last upload | 2019-08-20 |
| Time since last activity at time of writing | 6+ years |
| Description on PyPI | empty |

The package has had no releases, no bug fixes, no documentation updates,
and no metadata updates in over 6 years. PyPI policy considers this
strong evidence of abandonment.

## Process (per PyPI policy)

PyPI's project-name reassignment policy is documented at
[pypi.org/help/#project-name](https://pypi.org/help/#project-name). The
required steps, in order:

### Step 1 — contact the existing maintainer (mandatory)

PyPI requires a good-faith attempt to reach the current owner of the
name and wait at least 6 weeks for a response before escalating.

Contact channels for `puppet-master`'s owner:

- GitHub: https://github.com/qedsoftware (open an issue on the
  `puppet-master` repository if one exists, otherwise on their org
  page or a profile-level repo)
- Their existing project's repo: https://github.com/qedsoftware/puppet-master

Suggested outreach message (copy/paste, customize):

> Hi — I maintain [Puppetmaster](https://github.com/professorpalmer/Puppetmaster),
> an open-source agent orchestrator for Cursor / Codex / Claude Code, MIT-licensed.
> I noticed your `puppet-master` PyPI project hasn't had a release since
> August 2019 and is currently empty / placeholder-only. I'd love to
> publish my project as `puppetmaster` on PyPI, but PyPI's name
> normalization treats `puppetmaster` and `puppet-master` as the same
> name, so the upload is blocked.
>
> Would you be willing to release the `puppet-master` PyPI name? If
> yes, the PyPI process is for you to email admin@pypi.org confirming
> you're OK transferring it (or marking it as abandoned). I'm happy to
> draft the email for you and CC you on it, and I can also publish a
> redirect / deprecation notice under the old name to send anyone who
> still uses your package toward yours.
>
> If no, no worries at all — I'll continue publishing as
> `puppetmaster-ai`. Just thought it was worth asking.
>
> Thanks for considering, Cary

### Step 2 — escalate to PyPI admins (after 6 weeks of no response, OR with the maintainer's blessing)

Open an issue at [github.com/pypi/support](https://github.com/pypi/support)
titled `Project name reassignment request: puppetmaster`. Use the
template at:
https://github.com/pypi/support/blob/main/.github/ISSUE_TEMPLATE/name-request.yml

Required content (draft below — fill in dates of outreach attempts
before submitting):

> **Existing project**: https://pypi.org/project/puppet-master/
> **Project I'd like to publish under this name**: https://pypi.org/project/puppetmaster-ai/
> **GitHub repo**: https://github.com/professorpalmer/Puppetmaster
>
> **Why this qualifies under PyPI's reassignment policy**:
> 1. The existing project has had exactly one release (v0.1.0) on 2019-08-20 and zero activity in the 6+ years since.
> 2. The project's PyPI page has no description, no documentation links beyond the source repo, and the source repo itself has had no commits in years.
> 3. I have a substantive, actively-maintained, production-grade project at https://github.com/professorpalmer/Puppetmaster (currently v0.7.2, 199+ unit tests, validated end-to-end against Cursor / Codex / Claude Code / OpenAI APIs, MIT-licensed, with reproducible benchmark receipts). It is already publishing to PyPI as `puppetmaster-ai`.
> 4. Outreach attempts to the existing maintainer (qedsoftware on GitHub) on `<DATE>` and `<DATE>` have not received a response.
>
> **Mitigation for any existing users of `puppet-master`**: I'll publish a final v0.2.0 release of `puppet-master` (under whatever ownership transfer mechanism PyPI prefers) that imports nothing and emits a clear deprecation message pointing at the new project's PyPI page. Anyone with the old version pinned can keep using it; anyone upgrading sees a notice.
>
> Happy to provide any additional evidence requested.

### Step 3 — after reassignment

Once the name is granted:

1. Bump `pyproject.toml` `name = "puppetmaster-ai"` → `name = "puppetmaster"`.
2. Bump version (PyPI versions are immutable — pick the next semver, e.g. `0.7.3` or `0.8.0`).
3. Build + publish.
4. Replace the `puppetmaster-ai` PyPI listing with a thin metapackage
   that depends on `puppetmaster>=<version>` and prints a one-line
   migration notice on install. Don't yank the old `puppetmaster-ai`
   releases — that would break anyone with a pinned dependency.
5. Update README quickstart to use `pip install puppetmaster` and the
   v0.7.2 release notes on GitHub to note the rename.

## Tracking

- [ ] Outreach to qedsoftware sent (date: __________)
- [ ] Maintainer response received OR 6 weeks elapsed (date: __________)
- [ ] PyPI support issue filed (issue URL: __________)
- [ ] PyPI admins responded (date: __________)
- [ ] Name reassigned (date: __________)
- [ ] First release under bare `puppetmaster` name published (version: __________)
- [ ] `puppetmaster-ai` metapackage shim published

Nothing here is blocking. The project ships and is installable as
`pip install puppetmaster-ai` today.
