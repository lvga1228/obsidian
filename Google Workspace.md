---
category: "productivity"
skill_id: "google-workspace"
display_name: "Google Workspace"
---

# Google Workspace

**Skill ID:** `google-workspace`  
**Category:** [[productivity]]  
**Description:** Gmail, Calendar, Drive, Docs, Sheets via gws CLI or Python.

## Skill Content

# Google Workspace

Gmail, Calendar, Drive, Contacts, Sheets, and Docs — through Hermes-managed OAuth and a thin CLI wrapper. When `gws` is installed, the skill uses it as the execution backend for broader Google Workspace coverage; otherwise it falls back to the bundled Python client implementation.

## References

- `references/gmail-search-syntax.md` — Gmail search operators (is:unread, from:, newer_than:, etc.)
- `references/china-network.md` — Proxy quirks, pip mirrors, and Google API connectivity from within China

## Scripts

- `scripts/setup.py` — OAuth2 setup (run once to authorize)
- `scripts/google_api.py` — compatibility wrapper CLI. It prefers `gws` for operations when available, while preserving Hermes' existing JSON output contract.

## First-Time Setup

### When to choose Google Workspace over Himalaya

- **Chinese networks**: Gmail IMAP/SMTP ports are blocked. The Gmail REST API (HTTPS) works through HTTP proxies. If you're in China and need Gmail, Google Workspace is the only path that works out of the box.
- **Calendar / Drive / Sheets / Docs needed**: Himalaya is email-only. Google Workspace covers the full suite.
- **Email only, outside China**: Himalaya is simpler (App Password, 2-minute setup) and preferred.

The setup is fully non-interactive — you drive it step by step so it works
on CLI, Telegram, Discord, or any platform.

Define a shorthand first:

```bash
GSETUP="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/setup.py"
```

### Step 0: Check if already set up

```bash
$GSETUP --check
```

If it prints `AUTHENTICATED`, skip to Usage — setup is already done.

### Step 1: Triage — ask the user what they need

Before starting OAuth setup, ask the user TWO questions:

**Question 1: "What Google services do you need? Just email, or also
Calendar/Drive/Sheets/Docs?"**

- **Email only, outside China** → Use the `himalaya` skill instead — it works
  with a Gmail App Password and takes 2 minutes. No Google Cloud project needed.
- **Email only, in China** → Continue with this skill. Gmail IMAP/SMTP ports are
  blocked by GFW. The Gmail REST API (HTTPS) works through HTTP proxies.
- **Email + other services** → Continue with this skill.

> **Note**: The current `setup.py` always requests ALL scopes (gmail, calendar,
> drive, contacts, spreadsheets, docs). The `--services` flag is not yet
> implemented. The user will see all permissions on the consent screen, but only
> the APIs you actually call will be used.

**Question 2: "Does your Google account use Advanced Protection (hardware
security keys required to sign in)? If you're not sure, you probably don't
— it's something you would have explicitly enrolled in."**

- **No / Not sure** → Normal setup. Continue below.
- **Yes** → Their Workspace admin must add the OAuth client ID to the org's
  allowed apps list before Step 4 will work. Let them know upfront.

### Step 2: Create OAuth credentials (one-time, ~5 minutes)

Tell the user:

> You need a Google Cloud OAuth client. This is a one-time setup:
>
> 1. Create or select a project:
>    https://console.cloud.google.com/projectselector2/home/dashboard
> 2. Enable the required APIs from the API Library:
>    https://console.cloud.google.com/apis/library
>    Enable: Gmail API, Google Calendar API, Google Drive API,
>    Google Sheets API, Google Docs API, People API
> 3. Create the OAuth client here:
>    https://console.cloud.google.com/apis/credentials
>    Credentials → Create Credentials → OAuth 2.0 Client ID
> 4. Application type: "Desktop app" → Create
> 5. If the app is still in Testing, add the user's Google account as a test user here:
>    https://console.cloud.google.com/auth/audience
>    Audience → Test users → Add users
> 6. Download the JSON file and tell me the file path
>
> Important Hermes CLI note: if the file path starts with `/`, do NOT send only the bare path as its own message in the CLI, because it can be mistaken for a slash command. Send it in a sentence instead, like:
> `The JSON file path is: /home/user/Downloads/client_secret_....json`

Once they provide the path:

```bash
$GSETUP --client-secret /path/to/client_secret.json
```

If they paste the raw client ID / client secret values instead of a file path,
write a valid Desktop OAuth JSON file for them yourself, save it somewhere
explicit (for example `~/Downloads/hermes-google-client-secret.json`), then run
`--client-secret` against that file.

### Step 3: Get authorization URL

```bash
$GSETUP --auth-url
```

This prints the OAuth URL to stdout and also saves it to
`~/.hermes/google_oauth_last_url.txt`. The current script always requests all
available scopes (gmail, calendar, drive, contacts, spreadsheets, docs) — the
user will see these on the consent screen. Only APIs you actually call will be
used.

Agent rules for this step:
- `--auth-url` prints the raw URL directly to stdout (not JSON). Send it as-is to the user.
- **\"Google hasn't verified this app\" warning**: After the test user is added, the first authorization attempt still shows a warning page (`accounts.google.com/signin/oauth/warning`). Tell the user: click **\"Advanced\"** (高级), then **\"Go to Hermes Email (unsafe)\"** (前往…不安全). This is normal for apps in Testing mode and only appears once per account.
- Tell the user that the browser will fail on `http://localhost:1` after approval, and that this is expected.
- Tell them to copy the ENTIRE redirected URL from the browser address bar.
- If the user gets `Error 403: access_denied`, send them directly to `https://console.cloud.google.com/auth/audience` to add themselves as a test user.

### Step 4: Exchange the code

The user will paste back either a URL like `http://localhost:1/?code=4/0A...&scope=...`
or just the code string. Either works. The `--auth-url` step stores a temporary
pending OAuth session locally so `--auth-code` can complete the PKCE exchange
later, even on headless systems:

```bash
$GSETUP --auth-code "THE_URL_OR_CODE_THE_USER_PASTED"
```

If `--auth-code` fails because the code expired, was already used, or came from
an older browser tab, it now returns a fresh `fresh_auth_url`. In that case,
immediately send the new URL to the user and have them retry with the newest
browser redirect only.

### Step 5: Verify

```bash
$GSETUP --check
```

Should print `AUTHENTICATED`. Setup is complete — token refreshes automatically from now on.

### Notes

- Token is stored at `~/.hermes/google_token.json` and auto-refreshes.
- Pending OAuth session state/verifier are stored temporarily at `~/.hermes/google_oauth_pending.json` until exchange completes.
- If `gws` is installed, `google_api.py` points it at the same `~/.hermes/google_token.json` credentials file. Users do not need to run a separate `gws auth login` flow.
- To revoke: `$GSETUP --revoke`

## Usage

All commands go through the API script. Always set proxy first (required in China):

```bash
export https_proxy=http://127.0.0.1:15236 http_proxy=http://127.0.0.1:15236
GAPI="python $HOME/.hermes/skills/productivity/google-workspace/scripts/google_api.py"
```

### Gmail

```bash
# Search (returns JSON array with id, from, subject, date, snippet)
$GAPI gmail search "is:unread" --max 10
$GAPI gmail search "from:boss@company.com newer_than:1d"
$GAPI gmail search "has:attachment filename:pdf newer_than:7d"

# Read full message (returns JSON with body text)
$GAPI gmail get MESSAGE_ID

# Send (with optional attachment)
$GAPI gmail send --to user@example.com --subject "Hello" --body "Message text"
$GAPI gmail send --to user@example.com --subject "Hello" --body "Message text" --from '"Agent Name" <user@example.com>'

# Send with attachment (supports any file type — auto-detects MIME)
$GAPI gmail send --to user@example.com --subject "Report" --body "See attached" --attachment "/path/to/report.pdf"
$GAPI gmail send --to user@example.com --subject "Chart" --body "PNG attached" --attachment "/path/to/chart.png"
# Send
$GAPI gmail send --to user@example.com --subject "Hello" --body "Message text"
$GAPI gmail send --to user@example.com --subject "Report" --body "<h1>Q4</h1><p>Details...</p>" --html
$GAPI gmail send --to user@example.com --subject "Hello" --from '"Research Agent" <user@example.com>' --body "Message text"
$GAPI gmail send --to user@example.com --subject "Report" --body "See attached" --attachment ~/report.pdf
$GAPI gmail send --to user@example.com --subject "Photo" --body "" --attachment ~/photo.png

# Send with attachment
$GAPI gmail send --to user@example.com --subject "Report" --body "See attached" --attachment "/path/to/file.pdf"
$GAPI gmail send --to user@example.com --subject "Chart" --body "PNG attached" --attachment "/path/to/chart.png"

# Reply (automatically threads and sets In-Reply-To)
$GAPI gmail reply MESSAGE_ID --body "Thanks, that works for me."
$GAPI gmail reply MESSAGE_ID --from '"Support Bot" <user@example.com>' --body "Thanks"

# Labels
$GAPI gmail labels
$GAPI gmail modify MESSAGE_ID --add-labels LABEL_ID
$GAPI gmail modify MESSAGE_ID --remove-labels UNREAD
```

# Reply (automatically threads and sets In-Reply-To)
$GAPI gmail reply MESSAGE_ID --body "Thanks, that works for me."
$GAPI gmail reply MESSAGE_ID --from '"Support Bot" <user@example.com>' --body "Thanks"

# Labels
$GAPI gmail labels
$GAPI gmail modify MESSAGE_ID --add-labels LABEL_ID
$GAPI gmail modify MESSAGE_ID --remove-labels UNREAD
```

### Calendar

```bash
# List events (defaults to next 7 days)
$GAPI calendar list
$GAPI calendar list --start 2026-03-01T00:00:00Z --end 2026-03-07T23:59:59Z

# Create event (ISO 8601 with timezone required)
$GAPI calendar create --summary "Team Standup" --start 2026-03-01T10:00:00-06:00 --end 2026-03-01T10:30:00-06:00
$GAPI calendar create --summary "Lunch" --start 2026-03-01T12:00:00Z --end 2026-03-01T13:00:00Z --location "Cafe"
$GAPI calendar create --summary "Review" --start 2026-03-01T14:00:00Z --end 2026-03-01T15:00:00Z --attendees "alice@co.com,bob@co.com"

# Delete event
$GAPI calendar delete EVENT_ID
```

### Drive

```bash
$GAPI drive search "quarterly report" --max 10
$GAPI drive search "mimeType='application/pdf'" --raw-query --max 5
```

### Contacts

```bash
$GAPI contacts list --max 20
```

### Sheets

```bash
# Read
$GAPI sheets get SHEET_ID "Sheet1!A1:D10"

# Write
$GAPI sheets update SHEET_ID "Sheet1!A1:B2" --values '[["Name","Score"],["Alice","95"]]'

# Append rows
$GAPI sheets append SHEET_ID "Sheet1!A:C" --values '[["new","row","data"]]'
```

### Docs

```bash
$GAPI docs get DOC_ID
```

## Output Format

All commands return JSON. Parse with `jq` or read directly. Key fields:

- **Gmail search**: `[{id, threadId, from, to, subject, date, snippet, labels}]`
- **Gmail get**: `{id, threadId, from, to, subject, date, labels, body}`
- **Gmail send/reply**: `{status: "sent", id, threadId}`
- **Calendar list**: `[{id, summary, start, end, location, description, htmlLink}]`
- **Calendar create**: `{status: "created", id, summary, htmlLink}`
- **Drive search**: `[{id, name, mimeType, modifiedTime, webViewLink}]`
- **Contacts list**: `[{name, emails: [...], phones: [...]}]`
- **Sheets get**: `[[cell, cell, ...], ...]`

## Attachments

Send with `--attachment` flag (added by Hermes):

```bash
$GAPI gmail send --to user@example.com --subject "Report" --body "See attached" \
  --attachment "/path/to/file.pdf"
```

Supports PDF, PNG, CSV, Python files, etc. MIME type auto-detected from file extension.

## China Network Notes

- **Proxy required**: `export https_proxy=http://127.0.0.1:15236 http_proxy=http://127.0.0.1:15236` before all GAPI calls
- **pip dependencies**: Use Aliyun mirror without proxy (`-i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com`). pip + proxy = SSL errors.
- Gmail API works through HTTP proxy; IMAP/SMTP does NOT (GFW blocked)
- OAuth token at `~/.hermes/google_token.json` auto-refreshes

## Network Configuration (China-specific)

The Google API and pip dependencies have **opposite** network requirements:

| Operation | Proxy (15236) | Notes |
|-----------|--------------|-------|
| Google API calls (gmail send/search/etc.) | **ON** | `export https_proxy=http://127.0.0.1:15236` |
| OAuth token exchange | **ON** | |
| pip install google-api-python-client | **OFF** | Proxy causes SSL EOF errors with PyPI |
| pip install (alternative) | OFF + Aliyun mirror | `-i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com` |

### Verified install command

```bash
pip install google-api-python-client google-auth-oauthlib \
  -i https://mirrors.aliyun.com/pypi/simple \
  --trusted-host mirrors.aliyun.com
```

> **Do NOT use** `--services` flag with `setup.py` — it does not exist in the current version. All scopes are requested at once.

## Pitfalls (China Environment)

| Problem | Fix |
|---------|-----|
| `NOT_AUTHENTICATED` | Run setup Steps 2-5 above |
| `REFRESH_FAILED` | Token revoked or expired — redo Steps 3-5 |
| `HttpError 403: Insufficient Permission` | Missing API scope — `$GSETUP --revoke` then redo Steps 3-5 |
| `HttpError 403: Access Not Configured` | API not enabled — user needs to enable it in Google Cloud Console |
| `ModuleNotFoundError` | Run `$GSETUP --install-deps` |
| Advanced Protection blocks auth | Workspace admin must allowlist the OAuth client ID |
| **Google APIs unreachable (China)** | Set proxy: `export https_proxy=http://127.0.0.1:15236` |
| **pip install fails with SSL (China)** | Use Aliyun mirror: `pip install -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com` |
| **setup.py --services flag not supported** | The bundled setup.py does NOT accept `--services`. It always requests all scopes. The user must accept all permissions or use a custom setup. |
| **Gmail IMAP/SMTP blocked (GFW)** | Use Google Workspace REST API instead. Himalaya/Gmail IMAP will timeout on imap.gmail.com:993 and smtp.gmail.com:587. |
| **Browser OAuth redirect to localhost:1 fails** | Expected — this is normal. Copy the ENTIRE redirected URL from the address bar. |
| **access_denied during OAuth** | User not in test users → go to https://console.cloud.google.com/auth/audience → add email |
| **"未验证应用" warning** | Click "Advanced" → "Go to Hermes Email (unsafe)" → Allow |
| **pip SSL errors in China** | HTTP proxy breaks pip HTTPS. Use Aliyun mirror WITHOUT proxy: `pip install google-api-python-client google-auth-oauthlib -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com` |
| `setup.py --auth-url` freezes | It's auto-installing deps. Install them manually first (see above), then retry. Or run `$GSETUP --install-deps` with proxy off |
| Google API calls timeout | Need HTTP proxy for Google APIs in China: `export https_proxy=http://127.0.0.1:15236 http_proxy=http://127.0.0.1:15236` before running `$GAPI` commands |

## Revoking Access

```bash
$GSETUP --revoke
```

## China-Specific Notes

**Proxy requirements:**
- All Google API calls MUST go through Veee HTTP proxy: `export https_proxy=http://127.0.0.1:15236 http_proxy=http://127.0.0.1:15236`
- Gmail IMAP/SMTP ports (993/587) are BLOCKED by GFW — use this Gmail API skill instead
- pip install of dependencies: use Aliyun mirror WITHOUT proxy (`-i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com`)
- pip + proxy causes SSL errors (LibreSSL UNEXPECTED_EOF_WHILE_READING)

**OAuth setup quirks:**
- If user gets `Error 403: access_denied` (app not verified), they MUST add themselves as a test user at https://console.cloud.google.com/auth/audience
- After adding test user, regenerate auth-url — old URL becomes stale
- On the "Google hasn't verified this app" warning page, user clicks "Advanced" → "Go to Hermes Email (unsafe)"
