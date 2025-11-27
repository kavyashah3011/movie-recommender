const recommendBtn = document.getElementById("recommendBtn");
const movieInput = document.getElementById("movieInput");
const resultsDiv = document.getElementById("results");
const messageDiv = document.getElementById("message");

function setMessage(text, type = "info") {
    messageDiv.textContent = text || "";
    messageDiv.className = "message " + (type ? type : "");
}

async function fetchRecommendations() {
    const movie = movieInput.value.trim();
    if (!movie) {
        setMessage("Please type a movie name first.", "error");
        resultsDiv.innerHTML = "";
        return;
    }

    resultsDiv.innerHTML = "";
    setMessage("Loading recommendations…", "info");

    try {
        const res = await fetch("/recommend", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ movie })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            setMessage(errData.error || "Movie not found. Try another title.", "error");
            return;
        }

        const data = await res.json();
        const recs = data.recommendations;
        setMessage("");

        if (!recs || recs.length === 0) {
            setMessage("No recommendations found for this movie.", "info");
            return;
        }

        recs.forEach(rec => {
            const card = document.createElement("div");
            card.className = "card";

            card.innerHTML = `
                <img src="${rec.poster}" alt="${rec.title}">
                <div class="card-body">
                    <div class="card-title">${rec.title}</div>
                    <div class="card-meta">⭐ IMDb: ${rec.rating} &nbsp;&bull;&nbsp; 📅 ${rec.year}</div>
                    <div class="card-plot">${rec.plot}</div>
                </div>
            `;
            resultsDiv.appendChild(card);
        });

    } catch (err) {
        console.error(err);
        setMessage("Error fetching recommendations. Please try again.", "error");
    }
}

/* Click button */
recommendBtn.addEventListener("click", fetchRecommendations);

/* Press Enter in the input */
movieInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        fetchRecommendations();
    }
});
