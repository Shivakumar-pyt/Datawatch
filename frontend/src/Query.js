import React, { useState } from 'react';
import './background.css'
import MyNavbar from './MyNavbar';

export default function Query() 
{
  const defaultText = 'Enter the Query : ';
    const [value, onChangeText] = useState('Enter the Query : ');
  
    // If you type something in the text box that is a color, the background will change to that
    // color.

    const handleChange = () =>{
      return null
    }
    const handleClear = () => {
      onChangeText(defaultText);
    };

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
          onChange={(e) => onChangeText(e.target.value)}
          value={value}
          style={{ padding: 10, width: '100%', margin:'20px' }}
        />
      <div>
        </div>
        <button className='button1' onClick={handleClear}>
          Clear
        </button>
        <button className='button2' type='submit' onSubmit={handleChange}>
          Submit
        </button>
        </div>
      </div>
    );

}
  