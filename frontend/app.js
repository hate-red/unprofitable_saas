const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const messages = document.getElementById("messages");
const sendBtn = document.getElementById("send-btn");

function addMessage(text, className) {
  const div = document.createElement("div");
  div.className = "message " + className;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  addMessage(text, "user");
  input.value = "";
  sendBtn.disabled = true;

  const pending = addMessage("Думаю…", "assistant typing");
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const data = await response.json();

    pending.classList.remove("typing");
    pending.textContent = data.answer;
    if (data.sources && data.sources.length > 0) {
      const sources = document.createElement("div");
      sources.className = "sources";
      sources.textContent = "Источники: " +
        [...new Set(data.sources.map((s) => s.source))].join(", ");
      pending.appendChild(sources);
    }
  } catch (error) {
    pending.classList.remove("typing");
    pending.classList.add("error");
    pending.textContent = "Ошибка: " + error.message;
  } finally {
    sendBtn.disabled = false;
    input.focus();
    messages.scrollTop = messages.scrollHeight;
  }
});

// Enter — отправить, Shift+Enter — новая строка.
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
