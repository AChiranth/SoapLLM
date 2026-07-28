# Repository Guidelines

## Coding Style & Naming Conventions
Follow existing conventions per service. Python uses 4-space indentation, type hints where practical, and `snake_case`; tests are `test_*.py`. Format and lint python files with Ruff.

## Testing Guidelines
Prefer focused service-level tests before broad integration runs. When changing API or UI behavior, include the exact command run in your PR notes.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects, usually ticket-prefixed: `AUTH-6: rewrite verify handler` or `auth-9: remove magiclink and OTP`. Development is expected to happen in git worktrees under `.worktrees/`. The repository owner handles all `git add`, `git commit`, and `git push` operations; contributors and agents should leave staging, commits, and pushes to the owner unless explicitly asked otherwise. PRs should follow `.github/pull_request_template.md`: list created/modified/deleted files, add notes or follow-ups, and record testing results. Link the relevant issue/ticket, and include screenshots for frontend changes.

## Spec-first workflow
For non-trivial changes:
1. Do not start coding immediately.
2. First create or update `docs/specs/<task>.md`.
3. The spec should include but isn't limited to:
   - problem statement
   - goals / non-goals
   - constraints
   - relevant codepaths
   - proposed approach
   - risks / open questions
   - acceptance criteria
4. The format of the other documents in `docs/specs/` is a good format to follow
5. After the spec is written await user confirmation of it. Next, create `docs/plans/<task>.md`.
6. The plan should include:
   - ordered implementation steps
   - files likely to change
   - validation steps
   - rollback / risk notes
7. The format of the other documents in `docs/plans/` is a good format to follow. Tasks should be broken down into steps and all code-based steps should have associated code or pseudocode present.
8. After drafting the plan doc await user confirmation to continue to implementation phase. 
9. Implementation follows the order and tasks laid out in the plans document. When implementing, do one step at a time (a step is the sub-unit of a task) and ask for user confirmation before finalizing the step.

## Repo context
For a broader understanding of this project and its intentions, read `docs/Post_Training_Clinical_Scribe_Project_Plan_v2.md`.

## Custom Skills
If the user says "deep-review":
- Use the custom deep-review skill
- Follow the instructions in .codex/skills/deep-review/SKILL.md