const App = () => {
    const [selectedFile, setSelectedFile] = React.useState(null);
    const [preview, setPreview] = React.useState(null);
    const [prediction, setPrediction] = React.useState(null);
    const [loading, setLoading] = React.useState(false);

    const handleFileSelect = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreview(reader.result);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleSubmit = async () => {
        if (!selectedFile) return;

        setLoading(true);
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('http://localhost:5000/api/predict', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            setPrediction(data);
        } catch (error) {
            console.error('Error:', error);
            setPrediction({ error: 'Failed to process image' });
        } finally {
            setLoading(false);
        }
    };

    const DiseaseInfo = ({ info }) => (
        <div className="mt-4 space-y-3">
            <p className="text-gray-700">{info.description}</p>
            
            {info.symptoms.length > 0 && (
                <div>
                    <h3 className="font-semibold text-gray-800">Symptoms:</h3>
                    <ul className="list-disc list-inside text-gray-700">
                        {info.symptoms.map((symptom, index) => (
                            <li key={index}>{symptom}</li>
                        ))}
                    </ul>
                </div>
            )}
            
            {info.treatment.length > 0 && (
                <div>
                    <h3 className="font-semibold text-gray-800">Treatment:</h3>
                    <ul className="list-disc list-inside text-gray-700">
                        {info.treatment.map((step, index) => (
                            <li key={index}>{step}</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );

    return (
        <div className="min-h-screen bg-gray-100 py-6 flex flex-col justify-center sm:py-12">
            <div className="relative py-3 sm:max-w-xl sm:mx-auto">
                <div className="relative px-4 py-10 bg-white mx-8 md:mx-0 shadow rounded-3xl sm:p-10">
                    <div className="max-w-md mx-auto">
                        <div className="divide-y divide-gray-200">
                            <div className="py-8 text-base leading-6 space-y-4 text-gray-700 sm:text-lg sm:leading-7">
                                <h1 className="text-3xl font-bold text-center mb-8 text-green-600">
                                    Plant Disease Detector
                                </h1>
                                
                                <div className="flex flex-col items-center space-y-4">
                                    <input
                                        type="file"
                                        accept="image/*"
                                        onChange={handleFileSelect}
                                        className="hidden"
                                        id="file-upload"
                                    />
                                    <label
                                        htmlFor="file-upload"
                                        className="cursor-pointer bg-green-500 text-white px-4 py-2 rounded-md hover:bg-green-600 transition-colors"
                                    >
                                        Select Image
                                    </label>
                                    
                                    {preview && (
                                        <div className="mt-4">
                                            <img
                                                src={preview}
                                                alt="Preview"
                                                className="max-w-xs rounded-lg shadow-lg"
                                            />
                                        </div>
                                    )}
                                    
                                    {selectedFile && (
                                        <button
                                            onClick={handleSubmit}
                                            disabled={loading}
                                            className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition-colors disabled:bg-gray-400"
                                        >
                                            {loading ? 'Analyzing...' : 'Analyze Image'}
                                        </button>
                                    )}
                                    
                                    {prediction && (
                                        <div className="mt-4 p-6 bg-gray-50 rounded-lg w-full">
                                            {prediction.error ? (
                                                <p className="text-red-500">{prediction.error}</p>
                                            ) : (
                                                <div className="space-y-4">
                                                    <div className="border-b pb-4">
                                                        <h2 className="text-xl font-semibold text-gray-800">Results</h2>
                                                        <p className="text-gray-600">
                                                            Condition: <span className="font-semibold">{prediction.prediction.replace(/_/g, ' ')}</span>
                                                        </p>
                                                        <p className="text-gray-600">
                                                            Confidence: <span className="font-semibold">{(prediction.confidence * 100).toFixed(2)}%</span>
                                                        </p>
                                                    </div>
                                                    
                                                    {prediction.disease_info && (
                                                        <DiseaseInfo info={prediction.disease_info} />
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

ReactDOM.render(<App />, document.getElementById('root')); 