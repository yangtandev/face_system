## 10. Autonomous Decision-Making Guidelines
- **User Instructions as Iron Law:** Never simplify, alter, or "interpret" user instructions to fit your habits. If an instruction says "Do not upload X", it means exactly that.
- **Respect .gitignore:** The .gitignore file is an absolute boundary. Never use `git add -f` to bypass it unless explicitly instructed by the user for a specific file.
- **Verification over Assumption:** Before committing, always verify `git status`. Do not assume your actions aligned with your intent.

## 11. Operational Protocol (Strict Enforcement)
- **Mandatory Planning:** Upon receiving any instruction, you must first present a detailed, step-by-step plan and wait for the user's explicit confirmation before proceeding with any execution.
- **Explicit Permission Required:** You must obtain specific user approval before:
    1. Modifying `./handover_notes.md`.
    2. Creating any git commit.
    3. Pushing to remote repositories.

## 12. System Design Constraints (Iron Laws)
- **Zero False Acceptance:** The system must achieve 0% False Acceptance Rate (FAR). Any misidentification is unacceptable.
- **Single Enrollment Photo:** Each user is limited to exactly ONE enrollment photo. This photo cannot be replaced or augmented with additional angles. The system must solve challenges through algorithmic improvement, not data augmentation.
