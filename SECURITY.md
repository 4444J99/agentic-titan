# Security Policy

## Reporting Security Vulnerabilities

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, please use [GitHub Security Advisories](https://github.com/organvm-iv-taxis/agentic-titan/security/advisories/new)
to report vulnerabilities privately. This allows us to assess the risk and
prepare a fix before public disclosure.

If you are unable to use Security Advisories, you may email security concerns
to the repository maintainer directly.

### What to Include

Please include as much of the following information as possible:

- The type of issue (e.g., injection, authentication bypass, data exposure)
- Full paths of source file(s) related to the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

This information helps us triage and resolve the report quickly.

## Response Timeline

- **Acknowledgment:** Within 48 hours of receiving a report
- **Assessment:** Within 1 week of acknowledgment
- **Resolution:** Depends on severity; critical issues are prioritized

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |

## Security Best Practices for Contributors

1. **Never commit secrets** — API keys, passwords, tokens, or credentials
2. **Validate all inputs** — Sanitize user-supplied data
3. **Keep dependencies updated** — Review and apply security patches promptly
4. **Use HTTPS** — All external resources should be loaded over HTTPS
5. **Follow least privilege** — Request only the permissions you need

## Acknowledgments

We appreciate responsible disclosure and will credit reporters (with permission)
in our release notes.
