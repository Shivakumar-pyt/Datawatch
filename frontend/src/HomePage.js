import React, { useState } from 'react';
import './background.css';
import Background from './Background';
import Query from './Query';
import MyNavbar from './MyNavbar';
import DisplayRecords from './components/displayRecords';

function HomePage() {
  const [querySubmitted, setQuerySubmitted] = useState(false)
  return (
    <div className="container">
      <MyNavbar/>
      {<Background querySubmitted={querySubmitted}/>}
      <Query querySubmitted={querySubmitted} setQuerySubmitted={setQuerySubmitted}/>
    </div>
  );
}

export default HomePage;
