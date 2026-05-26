document.getElementById('loanForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    // 1. Gather values from form inputs
    const payload = {
        monthly_income: document.getElementById('monthly_income').value,
        age: document.getElementById('age').value,
        cibil_score: document.getElementById('cibil_score').value,
        existing_loan: document.getElementById('existing_loan').value
    };

    const container = document.getElementById('resultContainer');
    const valueText = document.getElementById('resultValue');
    const confidenceText = document.getElementById('resultConfidence');

    try {
        // 2. Transmit data to Flask API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.status === 'success') {
            // 3. Make UI update container visible
            container.classList.remove('hidden');
            valueText.innerText = data.prediction;
            confidenceText.innerText = `Model Match Confidence: ${data.confidence}`;

            // Adjust styles dynamically based on application evaluation
            if (data.prediction === "Approved") {
                container.className = "mt-6 p-4 rounded-xl border border-emerald-200 bg-emerald-50 text-center";
                valueText.className = "text-2xl font-extrabold my-1 text-emerald-700";
            } else {
                container.className = "mt-6 p-4 rounded-xl border border-rose-200 bg-rose-50 text-center";
                valueText.className = "text-2xl font-extrabold my-1 text-rose-700";
            }
        } else {
            alert("Error running inference: " + data.message);
        }

    } catch (error) {
        console.error("Transmission Error:", error);
        alert("Could not reach backend prediction server.");
    }
});