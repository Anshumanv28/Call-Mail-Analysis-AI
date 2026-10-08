# CLAUDE.md

Read `PROJECT_MEMORY.md` before working. It holds the project overview, which files are live, the context that is
not visible in the code, current accuracy, open items and the change log.

After any change to the project (code, models, data, config, docs), add a dated entry to the Change log in
`PROJECT_MEMORY.md` and fix any of its other sections that are no longer true. Git records what changed;
the log records why. Subagents you spawn must be told to do the same, or you must log their changes yourself.

@PROJECT_MEMORY.md
