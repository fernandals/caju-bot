// "Sair" (log out) where students can see it. Chainlit keeps a student
// logged in until the session expires (user_session_timeout in
// .chainlit/config.toml) and hides logging out in the avatar menu - on a
// shared lab computer the next student would carry on in the previous
// one's account. Two ways out, both Chainlit's own logout (POST /logout,
// which clears the auth cookie) followed by the login page:
// - a "Sair" button next to the avatar in the header;
// - any link to "#sair" in a message (the greeting's "Não é você? Sair").
(function () {
  var BUTTON_ID = "logout-button";

  function logout() {
    fetch("/logout", { method: "POST", credentials: "same-origin" }).finally(
      function () {
        window.location.href = "/login";
      }
    );
  }

  document.addEventListener(
    "click",
    function (event) {
      var link = event.target.closest && event.target.closest('a[href$="#sair"]');
      if (!link) return;
      event.preventDefault();
      event.stopPropagation();
      logout();
    },
    true
  );

  // The header is re-rendered by React (e.g. when the sidebar toggles),
  // so put the button back whenever it goes missing.
  function placeButton() {
    var avatar = document.getElementById("user-nav-button");
    if (!avatar || document.getElementById(BUTTON_ID)) return;

    var button = document.createElement("button");
    button.id = BUTTON_ID;
    button.type = "button";
    button.textContent = "Sair";
    button.title = "Sair da conta";
    button.addEventListener("click", logout);
    avatar.parentElement.insertBefore(button, avatar);
  }

  placeButton();
  new MutationObserver(placeButton).observe(document.documentElement, {
    childList: true,
    subtree: true,
  });
})();
