import React, { useState } from 'react';
import Login from '@react-login-page/base';

const MyLogin = () => {
  const [data, setData] = React.useState({});
  const [role, setRole] = useState("")
  const handle = async (even) =>  {


    even.preventDefault();
    const formData = new FormData(even.target);
    const data = Object.fromEntries(formData);
    setData({ ...data });
    console.log(data.username);


    try {
      const response = await fetch('http://127.0.0.1/8000/hackathon', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      })
      .then((res) => {
        return res.json();
      })
      .then((data) => {
        setRole(data["Role"])
      })
     
    } catch (error) {
      // Handle network errors
      console.error('Network error:', error);
    }
  };




  return (
    <div>
    <form onSubmit={handle}>
      <Login style={{ minHeight: 380 }} />
      <div>
        {role}
      </div>
    </form>
    </div>
  );
};

export default MyLogin;