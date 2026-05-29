# Puppetmaster demo GIF — "GIF as code".
#
# Render with VHS (https://github.com/charmbracelet/vhs):
#
#   brew install vhs           # or: go install github.com/charmbracelet/vhs@latest
#   vhs scripts/demo.tape      # writes docs/demo.gif
#
# Run from the repo root. The demo script needs no API keys.

Output docs/demo.gif

Set Shell bash
Set FontSize 16
Set Width 1280
Set Height 760
Set Padding 16
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set PlaybackSpeed 1.0

Hide
Type "clear" Enter
Show

Type "./scripts/demo.sh"
Sleep 600ms
Enter

# The script self-paces with DEMO_PAUSE between sections; give it room to finish.
Sleep 40s
