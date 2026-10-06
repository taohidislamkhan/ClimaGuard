# ClimaGuard — Viva notes: authentication & roles

Short answers in simple English, each followed by a Banglish explanation (🇧🇩).
Code references point to the files that implement each answer.

---

### 1. How are passwords stored?
We never store the password itself. On sign-up, Werkzeug's `generate_password_hash`
turns it into a **salted scrypt hash** (`scrypt:32768:8:1$<salt>$<hash>`), and only
that string goes into `users.password_hash`. At log-in, `check_password_hash` hashes
the typed password with the same salt and compares the results. scrypt is slow on
purpose and the salt is random per user, so a leaked database cannot be reversed
quickly, and two users with the same password get different hashes. Passwords are
never logged, not even failed ones (`dashboard/auth.py`, test `test_password_is_stored_hashed`).

🇧🇩 Amra password ta shoja database e rakhi na. Sign-up er shomoy password ke
scrypt diye "hash" kori, shathe ekta random salt. Database e shudhu oi hash thake.
Login er shomoy abar same bhabe hash kore milai. Hash theke asol password ber kora
khub kothin, ar salt er karone duijon er same password holeo hash alada hoy.

### 2. What is RBAC?
**Role-Based Access Control:** permissions are given to *roles*, and each user
has a role. ClimaGuard has three levels:
- **Guest:** only the regional pages.
- **User:** also their *own* My Risk, My Health, Settings and Account.
- **Admin:** also the `/admin` panel.

`@login_required` blocks guests (a login redirect, or a JSON 401 for `/api/*`), and
`@role_required("admin")` blocks normal users with a 403. On top of the role there is
an **object-level check**: every profile/settings query uses `current_user`, so user A
can never read or change user B's data, even by sending B's id (`dashboard/security.py`,
`tests/test_rbac.py`).

🇧🇩 RBAC mane "role diye access thik kora". Protita user er ekta role ase
(user ba admin), ar kon page/API ke use korte parbe ta role dekhe thik hoy.
Guest shudhu regional data dekhe, user nijer health data dekhe, admin admin panel
dekhe. Ar keu onner data dekhte pare na, karon shob query `current_user` diye filter hoy.

### 3. Why can't the admin see health data?
Health data (asthma, pregnancy, diabetes…) is sensitive personal data, and the admin
does not need it to do their job. Admins manage **accounts** (roles, locks,
resets), so admin pages and APIs return only account metadata: name, email, role,
status and dates. This follows **data minimisation** and **least privilege**: even
if an admin account is stolen, the attacker gets no health data. The profile is
read only for its owner, and a test scans every admin response for health fields
(`test_admin_responses_contain_no_health_data`).

🇧🇩 Health data khub private. Admin er kaj holo account manage kora (role,
lock, password reset), tar jonno karo asthma ba pregnancy jana dorkar nai. Tai
admin panel e shudhu nam, email, role, status dekhay. Admin account hack holeo
health data leak hobe na. Eta "least privilege" niyom: jar jeta dorkar, shudhu
setai dao.

### 4. How is CSRF prevented?
CSRF means another website tricks your browser into sending a request to ClimaGuard
with your cookies (for example "delete my profile"). Flask-WTF gives each session a
secret **CSRF token**:
- Every form has it as a hidden field.
- JavaScript sends it in the `X-CSRFToken` header on every POST/PUT/DELETE.
- The server rejects any state-changing request without a valid token (400).

Another site can't read the token, so it can't forge the request. Extra layers:
`SameSite=Lax` cookies, logout is POST-only, and a test checks that a POST without the token fails.

🇧🇩 CSRF holo onno website tomar browser ke diye lukiye ClimaGuard e request
pathay (tomar cookie shoho). Amra protita form ar API call e ekta gopon token
pathai, ja shudhu amader page jane. Token chara POST request ashle server reject
kore. Onno site oi token porte pare na, tai nokol request banate pare na.

### 5. Why is auth not part of the DVC pipeline?
DVC tracks things that must be **reproducible**: data → features → models →
metrics. Running `dvc repro` on the same inputs must give the same outputs.
User accounts are the opposite: **live runtime state** that changes every time
someone signs up or logs in, and it is private (password hashes, health profiles).
So auth lives only in the serving layer (the Flask app), and `instance/app.db` is
git-ignored and **not** DVC-tracked. Putting it in DVC would push private data to
the remote and break reproducibility. That is why `dvc status` still says "Data and pipelines are up to date."

🇧🇩 DVC er kaj holo ML pipeline reproducible rakha: same data dile same model.
Kintu user account protidin bodlay (notun sign-up, login), ar eta private data.
Eta model banate lage na, shudhu dashboard chalate lage. Tai auth DVC er baire,
`instance/app.db` Git ba DVC kothao jay na, ar pipeline ager motoi "up to date" thake.
