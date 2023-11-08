var content = document.documentElement.outerHTML;

console.log(window.location.toString());
function logCurrentUrl() {
  const url = window.location.toString();
  chrome.runtime.sendMessage({ url });
}
const url1 = window.location.toString();

logCurrentUrl();
const url = "http://localhost:5000/save";

fetch(url, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    address: url1,
    html: content,
  }),
}).then((response) => response.text());
