# AES Journal survey analysis

Analysis code for the extended JAES paper. The survey data is not included.

Setup:
* python3 -m venv aes-env
* source aes-env/bin/activate
* pip install -r requirements.txt
* python SurveyAnalysis.py

In all the scripts please change the  location of the files to you own local folders. 


# Codebook for the open-ended survey questions

This codebook describes how participants' answers to the three open-ended survey questions were coded for the paper. The raw survey data is not included in this repository, to protect participants' privacy.

## Coding procedure

- **Approach:** content analysis. Codes were developed inductively from the responses, then applied to every answer.
- **Codes per answer:** each response could receive up to two codes (columns `Code` and `Code2` in the coded files), as one answer often mentioned more than one idea.
- **Languages:** answers in English and Spanish were coded using the same codebook.
- **Blank answers:** left uncoded and excluded from the percentages for that question.
- **Reporting:** results are reported as the percentage of participants who answered each question (one participant can contribute to more than one code, so percentages can add up to more than 100%).
- **Label harmonisation:** small spelling and capitalisation differences in the coded files are standardised in the analysis script (the `fix` dictionaries in `SurveyAnalysis.py`).
- **Statistical tests:** the four most frequent codes of each question were compared across career stages and media types.
- **Themes:** each code is mapped to one of the five themes from our previous work: *Context*, *Workflow*, *Potential*, *Risks* and *Right Use*.

---

## 1. Workflow bottlenecks

*Survey question: the most time-consuming or challenging part of the participant's sound design workflow.*

| Code | Definition | Theme |
|---|---|---|
| Searching for sounds | Finding suitable sounds in libraries or online: browsing, keyword search, finding the right sample. | Workflow |
| Creating sounds | Designing sounds from scratch, e.g. layering, building textures, synthesis, designing sounds for specific events. | Workflow |
| Editing and tracklaying | Cleaning, editing, aligning and syncing audio, and laying tracks in a session. | Workflow |
| Client feedback and revisions | Changes requested by clients or directors, approvals and back-and-forth revisions. | Context |
| Mixing | Balancing and mixing sounds, including achieving a consistent mix across a project. | Workflow |
| Recording | Field recording, Foley sessions and capturing source material. | Workflow |
| Access to studios | Access to studios, equipment or facilities. | Context |
| QC and delivery | Quality control, exports, versioning and preparing deliverables (including localisation). | Workflow |
| Deadlines | Time pressure and tight schedules. | Context |
| Organisation and file management | Managing folders, naming files and organising one's own sound library. | Workflow |
| Refining to fit the project | Fine-tuning a sound so it fits the creative and technical needs of a specific project. | Workflow |
| Music creation and arrangement | Composing, arranging and programming MIDI parts or backing tracks. | Workflow |
| Troubleshooting | Solving technical problems with software or hardware. | Workflow |
| Keeping up with resources | Discovering new libraries and tools, and keeping track of resources as they change. | Context |

---

## 2. Principal concerns about AI

*Survey question: the participant's main concerns about using AI in sound design.*

| Code | Definition | Theme |
|---|---|---|
| Sound quality | AI output that is low quality, generic or contains artefacts. | Risks |
| Copyright and plagiarism | How models are trained, consent and credit for original creators, ownership of outputs and plagiarism. | Right Use |
| Job displacement | Loss of jobs, fewer opportunities or lower pay for sound professionals. | Risks |
| Unskilling | Reliance on AI leading practitioners to lose, or never develop, the foundational skills of the craft. | Risks |
| Controllability | Difficulty controlling, editing or precisely shaping AI output. | Risks |
| Work efficiency | AI slowing work down, e.g. time spent fixing or adapting outputs. | Workflow |
| Environmental issues | The energy use and environmental cost of AI. | Right Use |
| No concerns | The participant stated they had no concerns. | – |

---

## 3. Desired AI support

*Survey question: what kind of AI tool, if any, the participant would like to support their workflow.*

| Code | Definition | Theme |
|---|---|---|
| Editing assistance | Help with editing tasks, e.g. cleaning, noise reduction, dialogue editing and tedious edits. | Potential |
| Library search and organisation | Searching, tagging, naming and cataloguing sound libraries, including search by sound similarity. | Potential |
| Admin and repetitive tasks | Session preparation, file naming, organising projects and other routine or batch tasks. | Potential |
| Control and iteration | Tools whose output can be controlled, edited and refined through iteration with the user. | Potential |
| Learning and accessibility | Help learning tools and techniques, or making creative tools more accessible. | Potential |
| Sync to picture | Analysing video to find events, spot sounds and sync them to picture. | Potential |
| Tool implementation | Integration with DAWs, game engines or middleware, including scripting and batch implementation. | Potential |
| Sound generation | Generating sounds, ambiences, voices or music (including answers mentioning "generative AI"). | Potential |
| Stem separation | Separating mixed audio into individual sources or stems. | Potential |
| No AI wanted | The participant does not want AI in their workflow. | – |
| Unsure | The participant was unsure or did not know. | – |

---

## Coded files

| File | Question | Text column | Code columns |
|---|---|---|---|
| `Bottleneck.csv` | Workflow bottlenecks | `Bottleneck` | `Code`, `Code2` |
| `PrincipalConcerns.csv` | Principal concerns about AI | `Principal Concerns` | `Code`, `Code2` |
| `UseOfAI.csv` | Desired AI support | `Use of AI currently` | `Code`, `Code2` |

The coded files are not included in this repository, as they contain participants' original answers.
