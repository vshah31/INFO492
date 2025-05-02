function getTweetText() {
  const tweetNode = document.querySelector('[data-testid="tweetText"]');
  if (tweetNode) {
    const tweetText = tweetNode.innerText;
    console.log("Extracted Tweet:", tweetText);
    checkForMisinformation(tweetText);
  }
}

function checkForMisinformation(text) {
  alert("Checking tweet: " + text);
}

document.addEventListener('click', () => {
  setTimeout(getTweetText, 500);
});
