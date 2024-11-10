// import logo from './logo.svg';
import './App.css';
// import Lottie from 'react-lottie'
// import { Player } from '@lottiefiles/react-lottie-player';
// import { Route, Routes, Router } from 'react-router-dom';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
// import { DotLottiePlayer } from '@dotlottie/react-player';
import MyLogin from './MyLogin';
import Background from './Background';

import Query from './Query'
import HomePage from './HomePage';
import Warnings from './components/warnings';

// import animationData from 'src/animations/anime1.json'

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

// export default App;

export default function App() {
  return (
    <div>
     <Router> 
      <Routes>
        {/* <Route path="/" element={<HomePage />}/> */}
        <Route path="/" element={<MyLogin />}/>
        <Route path="/query" element={<HomePage/>}/>
        <Route path="/warnings" element={<Warnings/>}/>
      </Routes>
      </Router> 
    </div>
  );


  
  }

{/* // import { Player } from '@lottiefiles/react-lottie-player';

// const App = () => { */}
{/* //   return (
//     <div className='container'>
//       <h1>Using Lottie with React JS ⚛️</h1>

//       <Player/>

//     </div>
//   )
// }

// export default App;






 */}
