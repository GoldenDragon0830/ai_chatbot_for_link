import { useState, useEffect } from "react";
import { ChatInput } from "@/components/custom/chatinput";
import { PreviewMessage } from "../../components/custom/message";
import { Header } from "@/components/custom/header";
import { message } from "@/interfaces/interfaces";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";


interface Chatbot {
  created_at: string;
  link: string;
  name: string;
  pinecone_index: string;
}


export function Chat() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<message[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [chatbots, setChatbots] = useState<Chatbot[]>([]);
  const [selectedChatbot, setSelectedChatbot] = useState<number>(0);
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    fetch("https://soundglide.com/backend/api/start/chatbot_list")
      .then(res => res.json())
      .then(data => {
        if (data.success && data.chatbots.length > 0) {
          setChatbots(data.chatbots);
          setSelectedChatbot(0); // Select first by default
        }
      })
      .finally(() => setLoading(false))
  }, []);

  // Fetch chat history when selectedChatbot changes
  useEffect(() => {
    if (chatbots.length === 0) return;
    const index_id = chatbots[selectedChatbot]?.pinecone_index;
    if (!index_id) return;
    setLoading(true)
    fetch(`https://soundglide.com/backend/api/start/history?index_id=${encodeURIComponent(index_id)}`)
      .then(res => res.json())
      .then(data => {
        if (data.success && Array.isArray(data.history)) {
          // Map backend history to your message format
          const historyMessages = data.history.map((item : any) => ({
            content: item.message,
            role: item.sender,
            id: String(item.id),
          }));
          setMessages(historyMessages);
        } else {
          setMessages([]);
        }
      })
      .catch(() => setMessages([]))
      .finally(() => setLoading(false))
  }, [selectedChatbot, chatbots]);

  function cleanAIResponse(response?: string) {
    if (!response) return "";
    return response.replace("data: ", "").trim();
  }

  function handleSubmit(text?: string) {
    const userText = text ?? question;
    if (!userText.trim()) return;
  
    setMessages(prev => [
      ...prev,
      { content: userText, role: "user", id: String(prev.length) },
      { content: "Loading...", role: "assistant", id: String(prev.length + 1) }
    ]);
    setQuestion("");
  
    const index = chatbots[selectedChatbot]?.pinecone_index;
    if (!index) return;
  
    fetch(`https://soundglide.com/backend/api/start/chat?message=${encodeURIComponent(userText)}&index=${encodeURIComponent(index)}`)
      .then(res => res.text())
      .then(text => {
        setMessages(prev => {
          const updated = [...prev];
          const lastAssistantIdx = updated.findIndex(
            (msg, i) => msg.role === "assistant" && i === updated.length - 1
          );
          if (lastAssistantIdx !== -1) {
            updated[lastAssistantIdx] = {
              ...updated[lastAssistantIdx],
              content: cleanAIResponse(text) || "No response from AI."
            };
          }
          return updated;
        });
      })
      .catch(() => {
        setMessages(prev => {
          const updated = [...prev];
          const lastAssistantIdx = updated.findIndex(
            (msg, i) => msg.role === "assistant" && i === updated.length - 1
          );
          if (lastAssistantIdx !== -1) {
            updated[lastAssistantIdx] = {
              ...updated[lastAssistantIdx],
              content: "Error: Failed to get AI response."
            };
          }
          return updated;
        });
      });
  }

  return (
    <div className="flex flex-row min-w-0 h-dvh bg-background">
      {/* Left sidebar for chatbot list */}
      <div className="w-64 bg-gray-100 border-r flex flex-col">
        <div className="p-4 font-bold">Chatbots</div>
        <ul>
          {chatbots.map((bot, idx) => (
            <li
              key={bot.pinecone_index}
              className={`p-4 cursor-pointer ${selectedChatbot === idx ? "bg-blue-200 font-semibold" : ""}`}
              onClick={() => setSelectedChatbot(idx)}
            >
              {bot.name}
            </li>
          ))}
        </ul>
        {/* Bottom area for Go to Home button */}
        <div className="mt-auto p-4">
          <Button
            className="w-full"
            onClick={() => navigate("/")} // If using Next.js
            // onClick={() => window.location.href = "/"} // If not using Next.js
          >
            Home
          </Button>
        </div>
      </div>
      {/* Main chat area */}
      <div className="flex flex-col min-w-0 flex-1">
        <Header />
        {loading && (
          <div className="absolute inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
            <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-500"></div>
          </div>
        )}
        
        <div className="flex flex-col min-w-0 gap-6 flex-1 overflow-y-scroll pt-4">
          {messages.map((message, index) => (
            <PreviewMessage key={index} message={message} />
          ))}
        </div>
        <div className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl">
          <ChatInput  
            question={question}
            setQuestion={setQuestion}
            onSubmit={handleSubmit}
            isLoading={false}
          />
        </div>
      </div>
    </div>
  );
}