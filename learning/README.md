# learning/

A diary of lessons learned on this project. One markdown file per lesson.

## Instruction to Claude

Add a new file in this directory whenever one of these is true:

- The user **explicitly asks** you to log a lesson.
- You **taught the user** a better way to do something and they adopted it.
- A **non-obvious tradeoff** came up that's worth being able to look up later.
- The user **changed their mind** about a tool or approach after a discussion.

**Do not log:**
- Routine decisions already captured in `CLAUDE.md` §11 (decision log).
- Trivial syntax tips or language features the user already knows.
- Approaches the user considered and rejected.

The diary is for genuine "aha" moments that would help future-you or future-user re-derive the reasoning without replaying the whole conversation.

## File naming

`YYYY-MM-DD-<short-kebab-slug>.md` — e.g. `2026-04-19-ruff-vs-legacy-linters.md`. Date first so `ls` sorts chronologically.

## Entry format

Every file uses the same shape:

```markdown
# <Short title — one line>

- **What I learned:** one or two sentences.
- **What I was doing before:** <omit this bullet if N/A> the prior approach.
- **Why the new way is better:** concrete advantages.
- **Tradeoffs / when the old way still wins:** <omit this bullet if none>.
```

Keep entries tight — if a lesson takes more than ~10 lines, it's probably a primer and belongs in a section of `CLAUDE.md` instead of here.
