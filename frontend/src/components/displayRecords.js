import "../display.css"


export default function DisplayRecords({records}) {
    return (
        <div className="displaytable">
            {records.map((record) => (
                    <div className="record">
                        {record.map((entry) => (
                            <div>{entry}</div>
                        ))}
                    </div>
                ))}
        </div>
    )
}