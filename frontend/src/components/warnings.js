import React from "react";
import { useState, useEffect } from "react";
import "../warnings.css"


export default function Warnings(props) {
    const [warnings,setWarnings] = useState([])
    const url = "http://127.0.0.1:8000/hackathon/retrieveWarnings/"

    useEffect(() => {
        fetch(url).then((res) => {
            return res.json()
        }).then((data) => {
            console.log(data)
            setWarnings(data)
        }).catch((err) => {
            console.log(err)
        })
    },[])

    return (
        <div className="warnings">
            <p style={{color: "red",fontSize: "20px",textAlign: "center"}}>Warnings</p>
            <br></br>
            <div className="warning">
                {warnings.map((warning) => (
                    <div>{warning.warning_id} &emsp;
                    {warning.user} &emsp;
                    {warning.timestamp} &emsp;
                    {warning.status} &emsp;
                    {warning.query}
                    </div>
                ))}
            </div>
        </div>
    )
}