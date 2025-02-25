// src/App.js
import React from 'react';
import './App.css'; // Ensure this imports your CSS
import Header from './components/Header'; // Your Header component
import Dashboard from './components/Dashboard';

function App() {
    return (
        <div>
            <Header /> {/* Your header navigation component */}
            <main>
                <Dashboard /> {/* Your main dashboard content */}
            </main>
        </div>
    );
}

export default App;