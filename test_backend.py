Claude finished the response

Help me troubleshoot using matplotlib through VSCode debugger on Ubuntu. I want to run in the debugger instead of interactive window. Right now I just see a black screen after plt.show() that will not close or respond. It queries to force quit which kills the debug thread.
12:31 PM

This is a common issue: when running under the VSCode debugger (not the interactive window), plt.show() needs a working GUI event loop, and the default backend often doesn't play nicely with the debugger's process handling on Ubuntu.

A few things to check/try, roughly in order of likelihood:

1. Check which backend is being used

Add this near the top of your script temporarily:
python

import matplotlib
print(matplotlib.get_backend())

If it prints something like agg or module://... that's non-interactive, that's your problem — it's trying to render a GUI with a backend that doesn't support it well, which is exactly the kind of thing that produces a frozen/black window.

2. Force a proper interactive backend

Add before importing pyplot:
python

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt5/PySide installed
import matplotlib.pyplot as plt

    TkAgg requires python3-tk installed at the OS level: sudo apt install python3-tk
    Qt5Agg requires pip install PyQt5 (or pyqtagg/pyside6 as alternates)

Try TkAgg first since it has the lowest setup overhead.

3. Make sure you're not on a headless/no-DISPLAY session

Run echo $DISPLAY in the same terminal VSCode is using. If it's empty, your debug session has no X display to draw to, which would absolutely cause a black, unresponsive window. This can happen with SSH remote sessions without X forwarding, or certain WSL/container setups.

4. Check your launch.json

If you have a custom debug configuration, make sure "console" is set to "integratedTerminal" rather than "internalConsole". GUI apps launched with output redirected to the internal console can behave very strangely with windowing.
json

{
    "name": "Python: Current File",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal"
}

5. Confirm debugpy version

Older ptvsd/debugpy versions had known issues with GUI event loops blocking the debug protocol thread, causing the exact "force quit kills debugger" symptom you describe. Update the Python extension and debugpy to latest:
bash

pip install --upgrade debugpy

Can you run python3 -c "import matplotlib; print(matplotlib.get_backend())" in your VSCode integrated terminal and tell me what it prints? That'll narrow down whether it's a backend issue or a display/session issue.

Was using Qt5Agg: screens were black
Switch to TkAgg: image does not appear following plt.show()

echo $DISPLAY
:0

"console" is "integratedTerminal"

debugpy 1.8.20

python3 -c "import matplotlib; print(matplotlib.get_backend())"
qtagg
12:36 PM
Diagnosed debugger-environment mismatch causing matplotlib backend conflicts

Good data points — let's narrow this down further.

First, the key isolation test: does the exact same script work fine when run normally in the terminal (python3 yourscript.py), outside the debugger entirely? If yes, this confirms it's debugger-specific (debugpy/event-loop interaction), not an environment/display issue. If it also hangs outside the debugger, it's not actually a debugger problem at all.

A few other things to check in parallel:

1. Multiple matplotlib/Python installs
You ran the backend check from the integrated terminal and got qtagg, but that may be a different interpreter than the one VSCode's debugger is launching. Check:
bash

which python3
python3 -c "import sys; print(sys.executable)"

...and compare to the interpreter VSCode shows in the bottom-left status bar / your launch.json's python setting if explicitly set. A mismatch (e.g. system Python vs a venv, one with apt-installed PyQt5 and one with pip-installed) is a classic cause of "works in terminal, dies in debugger."

2. Try forcing the Qt platform plugin
Sometimes XWayland/Qt plugin negotiation silently fails and produces a black/frozen surface:
bash

QT_QPA_PLATFORM=xcb python3 yourscript.py

or set this as an env var in launch.json:
json

"env": { "QT_QPA_PLATFORM": "xcb" }

3. Confirm plt.show() is blocking, not detached
python

plt.show(block=True)

explicitly, in case something earlier (interactive mode, an ion() call) is leaving it non-blocking and the debugger's main thread races past it.

4. Try justMyCode: false and subProcess: false in launch.json
json

{
    "name": "Python: Current File",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal",
    "justMyCode": false,
    "subProcess": false
}

This rules out debugpy's subprocess/thread injection interfering with the Qt/Tk main loop, which is a known category of bug.

5. Check for a stray breakpoint or debugpy.breakpoint() call
Unlikely given your description, but worth a quick scan — a hit breakpoint inside the GUI loop's callback can look exactly like a frozen black window.

Can you confirm: does it run fine outside the debugger? And what does sys.executable print compared to your VSCode interpreter selection?

write a sample file to check if it runs outside debugger. Also include launch.json
12:41 PM
Created 2 files, read a file

To use:

    Drop plot_test.py anywhere in your project.
    Put launch.json in your project's .vscode/ folder (merge with existing configs if you already have one — just add the "Python: plot_test.py" entry to your configurations array).
    Open plot_test.py, then:
        Run outside debugger: python3 plot_test.py in the integrated terminal
        Run inside debugger: select "Python: plot_test.py" config and hit F5

Compare behavior between the two. Also try toggling matplotlib.use("TkAgg") to "Qt5Agg" in the script to test both backends under the same conditions.

Report back:

    Does it work outside the debugger?
    Does it work inside the debugger with QT_QPA_PLATFORM=xcb set?
    What does sys.executable print in each case — same path both times?

That'll tell us whether this is a debugpy/event-loop issue, a Qt/Wayland issue, or an interpreter mismatch issue.
Plot test
PY 
Launch
Code · JSON 


Claude is AI and can make mistakes. Please double-check responses.
Plot test · PY
"""
Minimal matplotlib smoke test.
 
Run this two ways and compare behavior:
  1) Outside the debugger:  python3 plot_test.py
  2) Inside the debugger:   F5 in VSCode (using the launch config below)
 
Expected: a window appears with a simple sine wave, and closing the
window (or pressing Ctrl+C in the terminal) lets the script exit cleanly.
"""
 
import sys
import matplotlib
 
# Force backend explicitly so we know exactly what's being used.
matplotlib.use("TkAgg")  # change to "Qt5Agg" to test that backend instead
 
import matplotlib.pyplot as plt
import numpy as np
 
print("Python executable:", sys.executable)
print("Matplotlib backend:", matplotlib.get_backend())
 
x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)
 
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Backend: " + matplotlib.get_backend())
 
print("Calling plt.show()...")
plt.show(block=True)
print("plt.show() returned -- window was closed.")
 

