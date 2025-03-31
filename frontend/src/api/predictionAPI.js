export const getPredictions = async (formData) => {
  try {
    const response = await fetch("/api/predict/", {
      method: "POST",
      body: formData, // Send FormData directly
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const data = await response.json();
    return data[0]?.name;
  } catch (error) {
    console.error("Error fetching predictions:", error);
  }
};
