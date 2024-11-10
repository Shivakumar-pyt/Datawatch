import React, { useState } from 'react';
import './background.css'
import MyNavbar from './MyNavbar';
import DisplayRecords from './components/displayRecords';

export default function Query({querySubmitted,setQuerySubmitted}) 
{
  const defaultText = 'Enter the Query : ';
  const [value, setValue] = useState("");
  const url = "http://127.0.0.1:8000/hackathon/processQuery/"
  const [selectRows,setSelectRows] = useState([])
  const [result,setResult] = useState("")
  const [error,setError] = useState("")

  const role_name = localStorage.getItem("role_name")
  const user_id = localStorage.getItem("user_id")

  const handleChange = (e) =>{
    e.preventDefault()

    setQuerySubmitted(true)

    if(value.length > 0) {
        fetch(url,{
          method: 'POST',
          headers: {'Content-Type' : 'application/json'},
          body: JSON.stringify({query: value,role: role_name, user_id: user_id})
        }).then((res) => {
          return res.json();
        }).then((data) => {
          if(typeof(data)==="string") {
            if (data !== "Query successfully executed...") {
              setError(data)
              setSelectRows([])
              setResult("")
            }
            else{
              setResult(data)
              setSelectRows([])
              setError("")
            }
          }
          else {
            setSelectRows(data)
            setResult("")
            setError("")
          }
          
        }).catch((err) => {
          console.log(err)
        })
    }
  }

  const handleClear = (e) => {
    setValue("")
    setQuerySubmitted(false)
    setSelectRows([])
    setError("")
    setResult("")
  }

  const handleType = (e) => {
    setValue(e.target.value)
    setError("")
    setResult("")
    setSelectRows([])
  }


    return (
      <div>
      <div className='component2'
        style={{
          backgroundColor: value,
          alignItems : 'center',
          justifyContent : 'center',
          // boxSizing : 'border-box'
        }}>
        <textarea
          rows={4}
          maxLength={100}
          onChange={(e) => handleType(e)}
          placeholder='Enter the query'
          value={value}
          style={{ padding: 10, width: '100%', margin:'20px' }}
        />
      <div>
        </div>
        <button className='button1' onClick={(e) => handleClear(e)}>
          Clear
        </button>
        <button className='button2' type='submit' onClick={(e) => handleChange(e)}>
          Submit
        </button>
        </div>

        {selectRows.length > 0 && 
        
          <DisplayRecords records={selectRows}/>  
        }

        {result.length > 0 && <div style={{color: 'green',marginTop: "-50vh",marginLeft: "8%",fontSize: "24px"}}>{result}</div>}

        {/* {error.length > 0 && <div style={{color:'red'}}>{error}</div>} */}

        {error.length > 0 && <div style={{color: 'red',marginTop: "-50vh",marginLeft: "8%",fontSize: "24px"}}>{error}</div>}
      </div>
    );

}
  