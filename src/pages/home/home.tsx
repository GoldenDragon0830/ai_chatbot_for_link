import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";

const STEPS = [
    "Finding pages on your website",
    "Loading and analyzing content",
    "Splitting content into chunks",
    "Preparing the AI database",
    "Adding your content to the AI",
    "Finalizing your chatbot"
];


export function Home() {
    const navigate = useNavigate();
    const [website, setWebsite] = useState("");
    const [error, setError] = useState("");
    const [progress, setProgress] = useState(0); // 0 to STEPS.length
    const [progressMessages, setProgressMessages] = useState<string[]>([]);
    const [showProgress, setShowProgress] = useState(false);
    const [currentMessage, setCurrentMessage] = useState("");
    const [done, setDone] = useState(false);

    function extractIndexAndName(url: string) {
        // Remove protocol
        let clean = url.replace(/^https?:\/\//, "").replace(/^www\./, "");
        // Remove trailing slash
        clean = clean.replace(/\/$/, "");
        // Remove .com, .ai, .net, .org, etc.
        const domain = clean.split("/")[0];
        const index = domain.replace(/\.(com|ai|net|org|io|co|app|dev|site|xyz|info|biz|us|uk|ca|au|in|me|tech|store|online|website|space|fun|live|pro|shop|club|cloud|today|agency|solutions|systems|group|world|company|digital|media|network|services|studio|works|zone|center|consulting|design|events|finance|gallery|global|marketing|partners|software|team|ventures|capital|academy|care|city|clinic|coach|community|education|energy|engineer|engineering|estate|express|finance|fitness|games|health|hotel|law|life|ltd|management|market|money|news|partners|photography|press|school|security|support|technology|tips|tools|training|travel|university|vacations|ventures|watch|zone)$/, "");
        const chatbot_name = index.charAt(0).toUpperCase() + index.slice(1) + " v1.0";
        return { index, chatbot_name };
    }

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        setError("");
        setProgress(0);
        setProgressMessages([]);
        setShowProgress(true);
        setDone(false);

        const { index, chatbot_name } = extractIndexAndName(website.trim());

        const es = new EventSource(
            `http://85.209.93.93:4006/create_chatbot?website=${encodeURIComponent(website.trim())}&index=${encodeURIComponent(index)}&chatbot_name=${encodeURIComponent(chatbot_name)}`
        );

        // es.onopen = () => {
        //     // Send POST data via fetch, then listen via EventSource
        //     fetch("http://85.209.93.93:4006/create_chatbot", {
        //         method: "GET",
        //         headers: { "Content-Type": "application/json" },
        //         body: JSON.stringify({
        //             website: website.trim(),
        //             index,
        //             chatbot_name,
        //         }),
        //     });
        // };

        es.addEventListener("done", (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.success) {
                    setDone(true);
                    setShowProgress(false);
                    es.close();
                    navigate("/chat");
                }
            } catch (e) {
                
            }
        })
        es.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.message) {
                    setCurrentMessage(data.message);

                    // Find which step this message matches
                    const stepIdx = STEPS.findIndex((step) => data.message === step);
                    if (stepIdx !== -1) {
                        setProgress(stepIdx + 1);
                        setProgressMessages((prev) => {
                            const updated = [...prev];
                            updated[stepIdx] = data.message;
                            return updated;
                        });
                    }
                }
                if (data.success) {
                    setDone(true);
                    setShowProgress(false);
                    es.close();
                    navigate("/chat");
                }
                if (data.error) {
                    setError(data.error);
                    setShowProgress(false);
                    es.close();
                }
            } catch (e) {
                // ignore parse errors
            }
        };

        es.onerror = (err) => {
            setError("Streaming connection error.");
            setShowProgress(false);
            es.close();
        };
    };

    return (
        <div
            style={{
                minHeight: "100vh",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                background: "#fff",
                position: "relative",
            }}
        >
            {/* ... existing header and logo ... */}
            <img
                src="/src/assets/images/Header.png"
                alt="header"
                style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "auto",
                    zIndex: 1,
                }}
            />
            <div style={{ zIndex: 2, width: "100%", marginTop: 100, display: "flex", flexDirection: "column", alignItems: "center" }}>
                <h1 style={{ fontWeight: 700, fontSize: 36, textAlign: "center", marginBottom: 40 }}>
                    Create a custom Ai interactive chat for your company.
                </h1>
                <form
                    style={{ display: "flex", alignItems: "center", width: 600, maxWidth: "90%" }}
                    onSubmit={handleSubmit}
                >
                    <input
                        type="text"
                        placeholder="Submit your website"
                        value={website}
                        onChange={e => setWebsite(e.target.value)}
                        style={{
                            flex: 1,
                            padding: "20px 24px",
                            borderRadius: 40,
                            border: "1px solid #ddd",
                            fontSize: 24,
                            outline: "none",
                            marginRight: -55,
                            background: "#f9f9f9",
                        }}
                    />
                    <button
                        type="submit"
                        style={{
                            width: 48,
                            height: 48,
                            borderRadius: "50%",
                            background: "#ff8c2b",
                            border: "none",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            cursor: "pointer",
                            boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                        }}
                        disabled={showProgress}
                    >
                        <svg width="24" height="24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="5" y1="12" x2="19" y2="12" />
                            <polyline points="12 5 19 12 12 19" />
                        </svg>
                    </button>
                </form>
                {error && <div style={{ color: "red", marginTop: 16 }}>{error}</div>}
            </div>
            {showProgress && (
                <div
                    style={{
                        position: "fixed",
                        top: 0,
                        left: 0,
                        width: "100vw",
                        height: "100vh",
                        background: "rgba(0,0,0,0.3)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        zIndex: 9999,
                    }}
                >
                    <div
                        style={{
                            background: "#fff",
                            padding: 32,
                            borderRadius: 16,
                            boxShadow: "0 2px 16px rgba(0,0,0,0.15)",
                            minWidth: 400,
                            maxWidth: "90vw",
                        }}
                    >
                        <h2 style={{ fontWeight: 600, fontSize: 22, marginBottom: 16 }}>
                            Creating your chatbot...
                        </h2>
                        <div style={{ marginBottom: 16 }}>
                            <div
                                style={{
                                    height: 8,
                                    width: "100%",
                                    background: "#eee",
                                    borderRadius: 4,
                                    overflow: "hidden",
                                    marginBottom: 16,
                                }}
                            >
                                <div
                                    style={{
                                        width: `${(progress / STEPS.length) * 100}%`,
                                        height: "100%",
                                        background: "#3b82f6",
                                        transition: "width 0.3s",
                                    }}
                                />
                            </div>
                            <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                                {STEPS.map((step, idx) => (
                                    <li
                                        key={step}
                                        style={{
                                            display: "flex",
                                            alignItems: "center",
                                            marginBottom: 6,
                                            color: idx < progress ? "#22c55e" : "#888",
                                        }}
                                    >
                                        <input
                                            type="checkbox"
                                            checked={idx < progress}
                                            readOnly
                                            style={{ marginRight: 8 }}
                                        />
                                        <span>
                                            {progressMessages[idx] || step}
                                        </span>
                                    </li>
                                ))}
                            </ul>
                            <div style={{ marginTop: 16, color: "#555" }}>
                                {currentMessage}
                            </div>
                        </div>
                        <Button
                            className="w-full"
                            onClick={() => setShowProgress(false)}
                            disabled={!done}
                        >
                            Close
                        </Button>
                    </div>
                </div>
            )}
            {error && <div style={{ color: "red", marginTop: 16 }}>{error}</div>}
        </div>
    );
}