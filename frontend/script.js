const API_URL = "http://127.0.0.1:8000/predict";

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

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    if (!response.ok) throw new Error("Server error: " + response.status);

    const result = await response.json();

    document.getElementById("result").innerHTML =
      result.prediction === 1
        ? `<span style="color:red;">⚠️ Patient likely diabetic (Probability: ${result.probability})</span>`
        : `<span style="color:green;">✅ Patient not diabetic (Probability: ${result.probability})</span>`;

    // Progress bar fill
    const probabilityPercent = Math.round(result.probability * 100);
    document.getElementById("progress-fill").style.width = probabilityPercent + "%";

  } catch (error) {
    document.getElementById("result").innerHTML =
      `<span style="color:orange;">❌ Error: ${error.message}</span>`;
  }
});

document.getElementById("generate-sample").addEventListener("click", function() {
  const sample = {
    Pregnancies: Math.floor(Math.random() * 6),
    Glucose: Math.floor(Math.random() * (180 - 80) + 80),
    BloodPressure: Math.floor(Math.random() * (90 - 60) + 60),
    SkinThickness: Math.floor(Math.random() * (40 - 15) + 15),
    Insulin: Math.floor(Math.random() * (200 - 50) + 50),
    BMI: (Math.random() * (35 - 18) + 18).toFixed(1),
    DiabetesPedigreeFunction: (Math.random() * 1).toFixed(2),
    Age: Math.floor(Math.random() * (65 - 20) + 20)
  };

  document.getElementById("pregnancies").value = sample.Pregnancies;
  document.getElementById("glucose").value = sample.Glucose;
  document.getElementById("blood_pressure").value = sample.BloodPressure;
  document.getElementById("skin_thickness").value = sample.SkinThickness;
  document.getElementById("insulin").value = sample.Insulin;
  document.getElementById("bmi").value = sample.BMI;
  document.getElementById("dpf").value = sample.DiabetesPedigreeFunction;
  document.getElementById("age").value = sample.Age;

  document.getElementById("result").innerHTML =
    `<span style="color:blue;">ℹ️ Sample patient data generated. Click Predict to test.</span>`;
  document.getElementById("progress-fill").style.width = "0%";
});
