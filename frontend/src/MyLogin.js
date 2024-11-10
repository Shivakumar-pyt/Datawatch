import React, { useState, useEffect } from 'react';
import Login from '@react-login-page/base';
import { useNavigate } from 'react-router-dom';

const MyLogin = () => {
  const [username,setUsername] = useState("")
  const [password,setPassword] = useState("")
  const [role, setRole] = useState("")
  const [error,setError] = useState("")
  const navigate = useNavigate()


  const url = "http://127.0.0.1:8000/hackathon/login/"

  const handle = async (even) =>  {
    even.preventDefault();
    const formData = new FormData(even.target);
    const data = Object.fromEntries(formData);
    setUsername(data.username)
    setPassword(data.password)
  };


  useEffect(() => {
    if(username.length > 0) {
        fetch(url,{
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({username: username, password: password})
        }).then((res) => {
          return res.json();
        }).then((data) => {
          if(data.user_id && data.role_name) {
            localStorage.setItem("role_name",data.role_name)
            localStorage.setItem("user_id",data.user_id)
            navigate("/query")
          }
          else{
            setError(data)
          }
        }).catch((err) => {
          console.log(err)
        })
    }
  },[username])


  return (
    <div>
    <form onSubmit={handle}>
      <Login style={{ minHeight: 380 }} />
      <div>
        {role}
      </div>
    </form>
    {error.length > 0 && <div style={{marginTop: "4%", color: 'red',textAlign: "center",fontSize: "24px"}}>{error}</div>}
    </div>
  );
};

export default MyLogin;