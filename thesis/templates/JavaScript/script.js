/*fetch("http://localhost:5000/")
  .then((response) => response.text())
  .then((html) => {
    document.getElementById("html").innerHTML = html;
    console.log(html);
  })

chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  const url = message.url;

  console.log("hiiiier" + url);
  const url1 = "http://localhost:5000/saveURL";

  fetch(url1, {
    method: "POST",
    headers: {
      "Content-Type": "text/plain",
    },
    body: url,
  })
    .then((response) => response.text())
    .then((content) => console.log(content))
    .catch((error) => console.error(error));
});;*/

document
  .getElementById("search-form")
  .addEventListener("submit", function (event) {
    event.preventDefault();
    var input = document.getElementById("input").value;

    fetch("http://localhost:5000/search?input=" + input)
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        var resultsDiv = document.getElementById("search-results");
        resultsDiv.innerHTML = "";

        if (data.results.length > 0) {
          var resultsList = document.createElement("ul");
          data.results.forEach(function (result) {
            var listItem = document.createElement("li");
            var link = document.createElement("a");
            link.href = result;
            link.textContent = result;
            link.target = "_blank";
            listItem.appendChild(link);
            resultsList.appendChild(listItem);
          });
          resultsDiv.appendChild(resultsList);
        } else {
          var noResults = document.createElement("p");
          noResults.textContent = "No results found.";
          resultsDiv.appendChild(noResults);
        }
      });
  });
