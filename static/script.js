document.getElementById("send-btn").addEventListener("click", sendMessage);
document
  .getElementById("user-input")
  .addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      sendMessage();
    }
  });

function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();

  if (message) {
    addMessage(message, "user-message");
    input.value = ""; // Clear input field

    // Send message to the server
    fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ques: message }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.answer) {
          addMessage(data.answer, "ai-message");
        } else {
          addMessage(data.message || "Error: No response", "ai-message");
        }
      })
      .catch((err) => {
        console.error("Error:", err);
        addMessage("Error communicating with server.", "ai-message");
      });
  }
}

function addMessage(text, className) {
  const chatBox = document.getElementById("chat-box");
  const messageDiv = document.createElement("div");
  messageDiv.className = `chat-message ${className}`;
  messageDiv.innerText = text;
  chatBox.appendChild(messageDiv);
  chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the bottom
}
