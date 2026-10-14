// Chainlit loads a single custom_js file (.chainlit/config.toml); this one
// loads the app's scripts.
["/public/login-unmask.js", "/public/logout.js"].forEach(function (src) {
  var script = document.createElement("script");
  script.src = src;
  document.head.appendChild(script);
});
