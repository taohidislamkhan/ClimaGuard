/* Auth + account forms: show/hide password, inline validation before submit, "Forgot password?" note.
   The server validates again; these checks only save a round trip. */
const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

function setFieldError(input, msg) {
  const field = input.closest(".field");
  field.querySelectorAll(".err[data-client]").forEach((e) => e.remove());
  field.classList.toggle("invalid", !!msg);
  input.toggleAttribute("aria-invalid", !!msg);
  if (msg) {
    const e = document.createElement("div");
    e.className = "err";
    e.dataset.client = "1";
    e.textContent = msg;
    field.appendChild(e);
  }
}

function checkField(input, form) {
  const v = input.value;
  switch (input.name) {
    case "name": return v.trim() ? "" : "Enter your name.";
    case "email": return EMAIL_RE.test(v.trim()) ? "" : "Enter a valid email address.";
    case "password":
      if (!v) return "Enter a password.";
      if (form.dataset.validate === "loose") return "";   // log in / confirm: no strength rule
      return v.length >= 8 && /[A-Za-z]/.test(v) && /\d/.test(v) ? "" : "At least 8 characters, with at least one letter and one number.";
    case "confirm": return v && v === form.elements.password.value ? "" : "Passwords do not match.";
    default: return "";
  }
}

document.querySelectorAll("form[data-validate]").forEach((form) => {
  const inputs = [...form.querySelectorAll(".field input")];
  inputs.forEach((input) => {
    input.addEventListener("blur", () => { if (input.value) setFieldError(input, checkField(input, form)); });
    input.addEventListener("input", () => { if (input.closest(".field").classList.contains("invalid")) setFieldError(input, checkField(input, form)); });
  });
  form.addEventListener("submit", (ev) => {
    let first = null;
    inputs.forEach((input) => {
      const msg = checkField(input, form);
      setFieldError(input, msg);
      if (msg && !first) first = input;
    });
    if (first) { ev.preventDefault(); first.focus(); return; }
    if (form.dataset.confirm && !confirm(form.dataset.confirm)) ev.preventDefault();
  });
});

document.querySelectorAll("[data-pw-toggle]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const input = btn.parentElement.querySelector("input");
    const show = input.type === "password";
    input.type = show ? "text" : "password";
    btn.setAttribute("aria-label", show ? "Hide password" : "Show password");
    btn.innerHTML = `<i data-lucide="${show ? "eye-off" : "eye"}"></i>`;
    lucide.createIcons();
  });
});

const forgot = document.getElementById("forgot-btn");
if (forgot) {
  forgot.addEventListener("click", () => {
    const box = document.getElementById("forgot");
    box.classList.toggle("hidden");
    forgot.setAttribute("aria-expanded", String(!box.classList.contains("hidden")));
  });
}

lucide.createIcons();
