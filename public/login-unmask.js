// The login screen reuses Chainlit's native password field to collect the
// student's full name (relabeled "Nome completo" in the pt-BR translation).
// A name is not a secret, so masking it as dots is confusing. Chainlit
// renders a built-in show/hide toggle next to that field; this clicks it
// once on the login page so the name is visible as plain text by default.
(function () {
  function reveal() {
    var input = document.getElementById("password");
    if (!input || input.type !== "password") return false;

    // The show/hide toggle is the only <button> sitting next to the input
    // inside its wrapper (<div class="relative">).
    var toggle = input.parentElement && input.parentElement.querySelector("button");
    if (!toggle) return false;

    toggle.click();
    return input.type === "text";
  }

  if (reveal()) return;

  // The login form mounts asynchronously (React SPA), so watch for it.
  var observer = new MutationObserver(function () {
    if (reveal()) observer.disconnect();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });

  // Give up after a while (e.g. the user is already past the login page).
  setTimeout(function () {
    observer.disconnect();
  }, 20000);
})();
