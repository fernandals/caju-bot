// The app's front-end tweaks to Chainlit, loaded through custom_js in
// .chainlit/config.toml (one file: Chainlit loads a single script).

// ---------------- LOGIN PAGE ----------------
// Chainlit's login screen -> the app's own (public/login.html). Chainlit
// routes to /login itself: on a first visit, after logging out, or when
// the session expires - including in-page navigations, hence the
// history hooks. Runs before the page renders, so its form never shows.
(function () {
  var LOGIN_PAGE = "/public/login.html";

  function check() {
    if (window.location.pathname.replace(/\/+$/, "") === "/login") {
      window.location.replace(LOGIN_PAGE);
    }
  }

  ["pushState", "replaceState"].forEach(function (method) {
    var original = history[method];
    history[method] = function () {
      var result = original.apply(this, arguments);
      check();
      return result;
    };
  });

  window.addEventListener("popstate", check);
  check();
})();

// ---------------- CITATION PANEL ----------------
// Chainlit opens its side panel by itself whenever a reply brings new
// side elements - here, the PDF behind every citation chip - so most
// answers opened the first cited PDF unasked. Keep the panel for clicks
// only: when it opens or switches PDF with no click on a chip just
// before, close it again.
(function () {
  var CLICK_WINDOW_MS = 1500;
  var lastChipClick = 0;
  var lastTitle = null;

  document.addEventListener(
    "click",
    function (event) {
      if (event.target.closest && event.target.closest(".element-link")) {
        lastChipClick = Date.now();
      }
    },
    true
  );

  function close() {
    // Desktop: the panel's back arrow. Mobile: a dialog that closes on Escape.
    var back = document.querySelector("#side-view-title button");
    if (back) {
      back.click();
    } else {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    }
  }

  new MutationObserver(function () {
    var title = document.getElementById("side-view-title");
    var text = title ? title.textContent : null;

    if (text === lastTitle) return;
    lastTitle = text;

    if (text !== null && Date.now() - lastChipClick > CLICK_WINDOW_MS) {
      close();
    }
  }).observe(document.documentElement, { childList: true, subtree: true, characterData: true });
})();
