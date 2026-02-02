## 10. Autonomous Decision-Making Guidelines
- **User Instructions as Iron Law:** Never simplify, alter, or "interpret" user instructions to fit your habits. If an instruction says "Do not upload X", it means exactly that.
- **Respect .gitignore:** The .gitignore file is an absolute boundary. Never use `git add -f` to bypass it unless explicitly instructed by the user for a specific file.
- **Verification over Assumption:** Before committing, always verify `git status`. Do not assume your actions aligned with your intent.

## 11. Operational Protocol (Strict Enforcement)
- **Mandatory Planning:** Upon receiving any instruction, you must first present a detailed, step-by-step plan and wait for the user's explicit confirmation before proceeding with any execution.
- **Post-Commit Documentation Protocol:** After every successful `git commit`, you MUST actively ask the user: "Do you want to update `./handover_notes.md` with the details of these changes?" You may ONLY proceed with the update after receiving explicit permission.
- **Explicit Permission Required:** You must obtain specific user approval before:
    1. Modifying `./handover_notes.md`.
    2. Creating any git commit.
    3. Pushing to remote repositories.

## 12. System Design Constraints (Iron Laws)
- **Zero False Acceptance:** The system must achieve 0% False Acceptance Rate (FAR). Any misidentification is unacceptable.
- **Single Enrollment Photo:** Each user is limited to exactly ONE enrollment photo. This photo cannot be replaced or augmented with additional angles. The system must solve challenges through algorithmic improvement, not data augmentation.
- **No Defeatism:** Never declare "maintain status quo" or "impossible to fix" based on a single failure. Lazy resignation is strictly forbidden. You must exhaust all algorithmic possibilities (e.g., hybrid scoring, dynamic thresholding, negative mining) before conceding.

## 13. Quality Assurance & Error Prevention (Self-Correction Protocol)
- **Logic Flow Verification:** Do not rely solely on syntax correctness or linter passes. You must mentally trace the data flow of key variables (e.g., ensuring a `reason` string generated in one function is actually passed as an argument to the logging function).
- **Context-Aware Code Review:** When using the `code-review` skill, explicitly instruct the reviewer to check against the *specific business requirements* (e.g., "Verify that the filename includes the failure reason"), not just generic bugs.
- **Call-Site Verification:** When modifying a function's signature (e.g., adding a parameter), you MUST grep/search for ALL call sites and verify they have been updated to pass the new argument correctly. Relying on default values (`foo="Unknown"`) often masks logic errors.
- **Slow Down & Double Check:** Prioritize correctness over speed. Before confirming a task is done, perform a line-by-line diff review of your own changes, specifically looking for "what I forgot to change" rather than just "what I changed".
- **Code-First Verification:** Before making any hypothesis or writing any simulation code, you MUST thoroughly read the relevant source code (function bodies, variable definitions). Assume the answer is already in the code, and do not waste time on invalid assumptions.
- **Date Verification:** Before editing `./handover_notes.md`, you MUST check the system time (`date` command) to ensure the documented date is correct.
