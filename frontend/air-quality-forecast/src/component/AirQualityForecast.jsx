import React, { useState } from "react";
import "./AirQualityForecast.css";
import { FaCalendarAlt } from "react-icons/fa";

const fetchForecast = async (location, date) => {
  try {
    const response = await fetch("http://localhost:5000/api/forecast", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ city: location, date }),
    });
    if (!response.ok) {
      throw new Error("Forecast not found.");
    }
    const result = await response.json();
    return result.forecast;
  } catch (error) {
    console.error("Error fetching forecast:", error);
    return null;
  }
};

const locations = ["Peenya", "SilkBoard", "BapujiNagar", "Hombegowda"];

export default function AirQualityForecast() {
  const [selectedLocation, setSelectedLocation] = useState("");
  const [selectedDate, setSelectedDate] = useState("");
  const [forecastData, setForecastData] = useState(null);
  const [error, setError] = useState("");
  const [show2024Plot, setShow2024Plot] = useState(false);
  const [showDaywisePlot, setShowDaywisePlot] = useState(false);
  const [showAllPlots, setShowAllPlots] = useState(false);

  const handleGetForecast = async () => {
    if(!selectedDate && selectedLocation){
      alert("📅 Please select a date");
      setShow2024Plot(false);
      return;
    }
    else if(!selectedLocation && selectedDate){
      alert("📍 Please select a location");
      setShow2024Plot(false);
      return;
    }
    if (!selectedLocation || !selectedDate) {
      alert("⚠️ Please select both location and date.");
      setShow2024Plot(false);
      return;
    }
    

    const data = await fetchForecast(selectedLocation, selectedDate);
    if (data && Object.keys(data).length > 0) {
      setForecastData(data);
      setError("");
      setShow2024Plot(false);
      setShowDaywisePlot(false);
      setShowAllPlots(false);
    } else {
      setForecastData(null);
      setError("No forecast data available for this date and location.");
      setShow2024Plot(false);
      setShowDaywisePlot(false);
      setShowAllPlots(false);
    }
  };
  const getMonthName = (dateStr) => {
    const dateObj = new Date(dateStr);
    return dateObj.toLocaleString("default", { month: "long" });
  };

  return (
    <div className="container">
      <div className="title-container">
      <div class="title-wrapper"> <h1 class="title">Air Pollution Forecast</h1></div>
                <p className="description">
                Worried about tomorrow's air? Let's forecast it! 🌬️
                Get future air quality alerts for Bangalore's most polluted zones🚦🌫️
                </p>
      </div>
      <div className="overlay">
        <h2>Select Forecast Criteria</h2>
        <div className="dropdown-container">
          <select
            className="dropdown"
            value={selectedLocation}
            onChange={(e) => setSelectedLocation(e.target.value)}
          >
            <option value="">Select Location</option>
            {locations.map((location) => (
              <option key={location} value={location}>
                {location}
              </option>
            ))}
          </select>

          <div className="date-picker-container">
            <input
              type="date"
              className="date-input"
              min="2025-03-01"
              max="2025-07-31"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
            />
          </div>
        </div>

        <button className="fetch-button" onClick={handleGetForecast}>
          Get Forecast
        </button>
  <div className="plot-button-container">
    <button
      className="plot-button"
      onClick={() => {
        if (!selectedLocation) {
          alert("Please select a location to view 2024 plot.");
          return;
        }
        setShow2024Plot(true);
        setShowDaywisePlot(false);
        setShowAllPlots(false);
      }}
    >
      Show 2024 Plot
    </button>
    <button
      className="plot-button"
      onClick={() => {
        if (!selectedLocation || !selectedDate) {
          alert("Please select both location and date for Day-Day plot.");
          return;
        }
        setShow2024Plot(false);
        setShowDaywisePlot(true);
        setShowAllPlots(false);
      }}
    >
      Day-Day Plot
    </button>
    <button
            className="plot-button"
            onClick={() => {
              if (!selectedDate) {
                alert("Please select a date to view all daywise plots.");
                return;
              }
              setShow2024Plot(false);
              setShowDaywisePlot(false);
              setShowAllPlots(true);
            }}
          >
            Show All Plots
          </button>
  </div>
</div>
 
      {/* Forecast Result */}
{forecastData && (
  <div className="forecast-output">
    <h3 className="forecast-title">
      Forecast for {selectedLocation} on {selectedDate}:
    </h3>
    <div className="forecast-card">
      {Object.entries(forecastData).map(([key, value]) => {
        const isAQICategory = key === "AQI_Category";
        const hoverClass = isAQICategory
          ? `aqi-category-${value?.toLowerCase().replace(/\s/g, "")}`
          : "";
        return (
          <div key={key} className={`forecast-item ${hoverClass}`}>
            <span className="forecast-key">{key}:</span>
            <span className="forecast-value">{value}
            </span>
          </div>
        );
      })}
    </div>
  </div>
)} 
 {/*  Overall AQI Display */}
 {forecastData && forecastData["AQI"] && forecastData["AQI_Category"] && (
        <div className={`overall-aqi-card aqi-category-${forecastData["AQI_Category"].toLowerCase().replace(/\s/g, "")}`}>
          <h3>🌍 Overall Air Quality</h3>
          <p className="aqi-value1">
            AQI: <strong>{forecastData["AQI"]}</strong>
          </p>
          <p className="aqi-category1">
            Category: <strong> {forecastData["AQI_Category"]}{forecastData["AQI_Category"] === "Good" && "🟢🌿 "}
    {forecastData["AQI_Category"] === "Satisfactory" && "🌤️🙂 "}
    {forecastData["AQI_Category"] === "Moderate" && "🌥️😐 "}
    {forecastData["AQI_Category"] === "Poor" && "🌫️😷 "}
   </strong>
          </p>
        </div>
      )}    
{/* 2024 Monthly Plot */}
  {show2024Plot && selectedLocation && (
    <div className="plot-image">
      <h4 style={{
          color: '#c6ff00',
          backgroundColor: 'rgba(0, 0, 0, 0.7)', 
          padding: '10px 20px',
          borderRadius: '12px',
          fontWeight: 'bold',
          fontSize: '1.4rem',
          border: '2px solid rgba(255, 255, 255, 0.9)',     
          boxShadow: '0 0 12px 3px rgba(255, 255, 255, 0.5)',
              }}>2024 Monthly Forecast for {selectedLocation}</h4>
          <img
            src={`/plots/monthly/${selectedLocation}.png`}
            alt={`2024 Plot for ${selectedLocation}`}
            className="plot-image-display"
          />
        </div>
      )}
{/* Daywise Plot */}
  {showDaywisePlot && selectedLocation && selectedDate && (
      <div className="plot-image">
        <h4 style={{
                color: '#c6ff00',
                backgroundColor: 'rgba(0, 0, 0, 0.7)', 
                padding: '10px 20px',
                borderRadius: '12px',
                fontWeight: 'bold',
                fontSize: '1.4rem',
                border: '2px solid rgba(255, 255, 255, 0.9)',     
                boxShadow: '0 0 12px 3px rgba(255, 255, 255, 0.5)',
                  }}>
          Daywise AQI for {selectedLocation} - {getMonthName(selectedDate)} 2025
          </h4 >
          <img
            src={`/plots/daywise/AQI_${selectedLocation}_${getMonthName(selectedDate)}.png`}
            alt={`Daywise AQI for ${selectedLocation} - ${getMonthName(selectedDate)}`}
            className="plot-image-display"
          />
        </div>
      )}
       {/* Show All City Plots in Selected Month */}
      {showAllPlots && selectedDate && (
        <div className="plot-image">
          <h4 style={{
                  color: '#c6ff00',
                  backgroundColor: 'rgba(0, 0, 0, 0.7)', 
                  padding: '10px 20px',
                  borderRadius: '12px',
                  fontWeight: 'bold',
                  fontSize: '1.4rem',
                  display: 'inline-block',
                  border: '2px solid rgba(255, 255, 255, 0.9)',     
                  boxShadow: '0 0 12px 3px rgba(255, 255, 255, 0.5)',
                    }}>Daywise AQI for All Cities - {getMonthName(selectedDate)} 2025</h4>
          <div className="all-plots-container">
            {locations.map((city) => (
              <div
              key={city}
              className="all-plot-image">
                <h5 className="city-name-box">{city}</h5>
                <img
                  src={`/plots/daywise/AQI_${city}_${getMonthName(selectedDate)}.png`}
                  alt={`Daywise AQI for ${city}`}
                  className="plot-image-display1"
                />
               </div>
            ))}
          </div>
          </div>
      )}
      {/* Error Message */}
      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}
    </div>
  );
}



