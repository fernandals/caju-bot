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
