import Lottie from 'react-lottie'
import  './background.css'


import { DotLottiePlayer } from '@dotlottie/react-player';

export default function Background({querySubmitted}) {
    return (
      <div className='animation'>

        {!querySubmitted && <DotLottiePlayer src="https://lottie.host/c4f632e3-0e6c-4e09-84e6-9ec35bdfabdd/RXGhMa2tzc.json" autoplay={true} loop/>}

      </div>
    );
    
}