import { useNavigate } from "react-router-dom";
import { useState } from "react";


export function Home() {
    const navigate = useNavigate();
    const [website, setWebsite] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

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

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError("");
        if (!website.trim()) {
            setError("Please enter your website link.");
            return;
        }
        setLoading(true);
        const { index, chatbot_name } = extractIndexAndName(website.trim());
        try {
            const res = await fetch("http://85.209.93.93:4006/create_chatbot", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    website: website.trim(),
                    index,
                    chatbot_name
                })
            });
            if (res.status === 200) {
                setLoading(false);
                navigate("/chat");
            } else {
                setLoading(false);
                setError("Failed to create chatbot. Please try again.");
            }
        } catch (e) {
            setLoading(false);
            setError("Network error. Please try again.");
        }
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
                        disabled={loading}
                    >
                        <svg width="24" height="24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="5" y1="12" x2="19" y2="12" />
                            <polyline points="12 5 19 12 12 19" />
                        </svg>
                    </button>
                </form>
                {error && <div style={{ color: "red", marginTop: 16 }}>{error}</div>}
            </div>
            {loading && (
                <div style={{
                    position: "fixed",
                    top: 0, left: 0, width: "100vw", height: "100vh",
                    background: "rgba(0,0,0,0.3)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    zIndex: 9999
                }}>
                    <div style={{
                        background: "#fff",
                        padding: 32,
                        borderRadius: 16,
                        boxShadow: "0 2px 16px rgba(0,0,0,0.15)",
                        fontSize: 20,
                        fontWeight: 500
                    }}>
                        Creating your chatbot, please wait...
                    </div>
                </div>
            )}
        </div>
    );
}