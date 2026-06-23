## **STUDiO: Source Translation Utility for Device-Independent Oscillation**

STUDiO is a Python Shiny app for editing, translating, and exporting stroboscopic stimulation sourcecode for RoXiva/RX1- and Lucia/LUCiO-style workflows.

The app combines a manual curve editor, audio analysis tools, an audio scrubber, and export utilities for generating editable stimulation sequences from either hand-drawn oscillator curves or audio-derived control parameters.


## **What STUDiO Does**

STUDiO is designed for researchers, artists, and device developers who need to inspect, edit, or translate stimulation parameters outside of device-native editors.


## **It currently supports:**

Manual editing of oscillator curves for frequency, duty cycle, and luminance

Four RX1-style oscillator channels: `OSC1`-`OSC4`

Optional `SUN`/halogen luminance editing for Lucia/LUCiO-style exports

Audio upload and playback with waveform/spectrogram preview

Audio-derived frequency, duty-cycle, and luminance extraction

FFT, Harmonic FFT, and CQT-based audio analysis modes

Musical interval transposition of extracted frequency candidates

Frequency folding into stroboscopic/RX1-safe ranges

Audio-to-oscillator application for one selected oscillator or all four oscillators

Export to RX1-compatible `.txt` sourcecode

Export to editable JSON project files

Export to preview PNG figures

Export to audio-analysis CSV files

Export to Lucia/LUCiO `.lscf` files using a validated template

Export of LUCiO debug CSV files for checking row-level values


**Relationship to LUCiO**

STUDiO is a companion tool to LUCiO: the Lucia Unified Control Interface for OctAVEs.

Where LUCiO focuses on offline audio-to-sourcecode generation, STUDiO focuses on interactive editing, translation, manual codification, and sourcecode inspection.


**In short:**

LUCiO = audio-derived Lucia/RX1-compatible sourcecode generation

STUDiO = interactive sourcecode editing, translation, and manual authoring


**Installation**

Clone this repository:
```bash
git clone https://github.com/YOUR-USERNAME/STUDiO.git
cd STUDiO
```
Create a virtual environment:
```bash
python -m venv .venv
```
Activate it.
On Windows:
```bash
.venv\Scripts\activate
```
On macOS/Linux:
```bash
source .venv/bin/activate
```
Install the requirements:
```bash
pip install -r requirements.txt
```
**Requirements**
The app currently depends on:
```text
shiny
numpy
pandas
scipy
matplotlib
soundfile
librosa
standard-aifc
standard-sunau
```
These are listed in `requirements.txt`.
**Running the App**
Run STUDiO locally with:
```bash
shiny run --reload app.py
```
On Windows, depending on your Python installation, you may prefer:
```bash
py -3.13 -m shiny run --reload app.py
```
Then open the local URL shown in the terminal.


**Recommended First Test**

For first tests, use a short `.wav` file and conservative audio-analysis settings.

Recommended starting settings:

Engine: `FFT Peaks`

Voices to Extract: `1` or `2`

Audio Step Duration: `0.2-0.5 s`

Frequency Mapping: `SLS Centre` or `Fold to RX1`

Duty Cycle: `Fixed` at 50%

Luminance: `Amplitude`

Longer audio files and dense time steps can create very large analysis tables and exports. For long songs, start with a coarser audio step before reducing the step duration.


**Basic Workflow**

A typical STUDiO workflow is:

Set a sequence name and total duration.

Draw or edit curves for oscillator frequency, duty cycle, and luminance.

Optionally upload an audio file and run audio analysis.

Overlay audio-derived frequency, duty, or luminance onto the editor.

Apply audio-derived values to one oscillator or map audio voices across all four oscillators.

Inspect the preview and export estimate.

Export the desired output format.


**Export Options**

STUDiO currently includes several export paths.

*RX1 TXT*

Exports an RX1-style text sourcecode using `TIM`, `DUR`, and `STP` rows. The app enforces an RX1 line-count limit before export.

*Editable JSON*

Exports the current project as an editable JSON file that can be reloaded into the app later.

*Preview PNG*

Exports the current preview plot as a PNG.

*Audio CSV*

Exports the audio-analysis table, including raw and mapped frequencies, amplitude, occupancy, harmonic-band values, duty cycle, and luminance.

*LUCiO / Lucia LSCF*

Exports a Lucia/LUCiO `.lscf` file using a validated dynamic-duty template. The app preserves the template header, patches row-level control values, and recomputes the final XOR checksum.

For SUN/halogen control, select:
```text
OSC = SUN
Parameter = Luminance
```

Then draw a 0-100 luminance curve before exporting the `.lscf` file.

*LUCiO Debug CSV*

Exports a row-level CSV showing the control values used to construct the `.lscf` file. This is useful for checking achieved oscillator frequencies, cycle counts, halogen values, and patched row values before device testing.


**Audio Analysis Modes**

STUDiO includes three audio-analysis engines:

FFT Peaks: fast spectral peak extraction; recommended for first tests.

Harmonic FFT: ranks candidates using energy at harmonic partials; useful for more tonally stable sources.

CQT Peaks: semitone-spaced analysis that can work well for pitched musical material, but may be slower.

Extracted frequencies can be folded into stroboscopic ranges, snapped to a musical grid, and transposed by musical interval before being applied to oscillator curves.


**Frequency Mapping**

The app includes several frequency-mapping options, including:

No mapping

Fold to stroboscopic range

Centre within an SLS-like range

Alpha-array style mapping

Fold to RX1-safe range

These mappings are intended to make audio-derived frequencies usable as stroboscopic control rates rather than direct audio-frequency values.


**File Structure**

```text
STUDiO/
├── app.py              # Main Python Shiny application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```


**Notes on Audio Files**

WAV is the safest format for initial local testing. MP3, FLAC, and OGG may work depending on your local audio backend and Python environment.

For long audio files, use a larger audio step duration at first. Very fine step durations can create large tables, dense plots, and long sourcecode exports.


**Safety Notice**

This software is intended for research and development workflows involving stroboscopic stimulation. Stroboscopic light can be uncomfortable or unsafe for some individuals, particularly people with photosensitive epilepsy or other neurological sensitivities.

Do not use generated or edited stimulation files with participants unless they have been reviewed, validated, and approved under the appropriate safety, ethics, and device-testing procedures.

Always validate exported files in the target device software before experimental or participant-facing use.


**Development Status**

STUDiO is under active development. File-format support, export behaviour, audio-analysis settings, and device compatibility may change as the workflow is refined.


**Licence**

to update
```text
MIT Licence for code.
Separate rights notices may apply to stimulation files, audio files, templates, device-specific sourcecode formats, questionnaires, and third-party assets.
```


**Acknowledgement**

STUDiO was developed as part of an inter-device stroboscopic stimulation workflow for translating, editing, and manually codifying oscillator/sourcecode parameters across research and device-development contexts.
