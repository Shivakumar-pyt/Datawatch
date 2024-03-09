// React component
import React, { useState } from 'react';

function MyComponent() {
  const [formData, setFormData] = useState({});

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Make HTTP POST request to Django backend
    try {
      const response = await fetch('http://your-django-backend/api/endpoint', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      });

      if (response.ok) {
        // Handle successful response from Django backend
        console.log('Data sent successfully');
      } else {
        // Handle error response from Django backend
        console.error('Failed to send data');
      }
    } catch (error) {
      // Handle network errors
      console.error('Network error:', error);
    }
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <form onSubmit={handleSubmit}>
      <input type="text" name="field1" onChange={handleChange} />
      <input type="text" name="field2" onChange={handleChange} />
      <button type="submit">Submit</button>
    </form>
  );
}

export default MyComponent;
