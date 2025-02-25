import React, { useState } from "react";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import { Button, Form } from "react-bootstrap";

const EventFilter = ({ onFilter, eventOptions = [] }) => {
    const [startDate, setStartDate] = useState(null);
    const [endDate, setEndDate] = useState(null);
    const [selectedEvent, setSelectedEvent] = useState("");
    const [compareReturns, setCompareReturns] = useState(false);
  
    const handleFilter = () => {
      onFilter({
        startDate,
        endDate,
        selectedEvent,
        compareReturns,
      });
    };
  
    return (
      <div className="mb-3 p-4 rounded-lg shadow-lg bg-white">
        <h3 className="text-lg font-bold mb-2">Filter Events</h3>
  
        {/* Date Range Filters */}
        <div className="flex space-x-4 mb-2">
          <DatePicker
            selected={startDate}
            onChange={(date) => setStartDate(date)}
            placeholderText="Start Date"
            className="p-2 border rounded"
          />
          <DatePicker
            selected={endDate}
            onChange={(date) => setEndDate(date)}
            placeholderText="End Date"
            className="p-2 border rounded"
          />
        </div>
  
        {/* Event Type Dropdown */}
        <div className="mb-2">
          <label className="block mb-1">Select Event Type</label>
          <select
            value={selectedEvent}
            onChange={(e) => setSelectedEvent(e.target.value)}
            className="p-2 border rounded w-full"
          >
            <option value="">All Events</option>
            {eventOptions.map((event, index) => (
              <option key={index} value={event}>
                {event}
              </option>
            ))}
          </select>
        </div>
  
        {/* Comparison Toggle */}
        <div className="flex items-center mb-3">
          <input
            type="checkbox"
            checked={compareReturns}
            onChange={() => setCompareReturns(!compareReturns)}
            className="mr-2"
          />
          <label>Compare Cumulative Returns</label>
        </div>
  
        {/* Apply Filter Button */}
        <Button variant="primary" onClick={handleFilter}>
          Apply Filter
        </Button>
      </div>
    );
  };
  
export default EventFilter;
