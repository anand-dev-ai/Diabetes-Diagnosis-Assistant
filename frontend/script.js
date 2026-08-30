document.getElementById("patient-form").addEventListener("submit", async function(e) {
  e.preventDefault();

  const data = {
    Pregnancies: parseInt(document.getElementById("pregnancies").value),
    Glucose: parseFloat(document.getElementById("glucose").value),
    BloodPressure: parseFloat(document.getElementById("blood_pressure").value),
    SkinThickness: parseFloat(document.getElementById("skin_thickness").value),
    Insulin: parseFloat(document.getElementById("insulin").value),
    BMI: parseFloat(document.getElementById("bmi").value),
    DiabetesPedigreeFunction: parseFloat(document.getElementById("dpf").value),
    Age: parseInt(document.getElementById("age").value)
  };

  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });

  const result = await response.json();
  document.getElementById("result").innerHTML =
    result.prediction === 1
      ? `<span style="color:red;">⚠️ Patient likely diabetic (Probability: ${result.probability})</span>`
      : `<span style="color:green;">✅ Patient not diabetic (Probability: ${result.probability})</span>`;
});
