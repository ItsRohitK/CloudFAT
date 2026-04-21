import boto3
import json
MONGODB_URI="mongodb+srv://aman11:Aman%234696@cluster0.czqqffc.mongodb.net/?appName=Cluster0"
runtime_client = boto3.client('sagemaker-runtime')
ENDPOINT_NAME = "sagemaker-xgboost-2026-04-20-19-16-27-692" 

def lambda_handler(event, context):
    try:
        # 1. Parse the input from Postman
        if 'body' in event:
            body = json.loads(event['body'])
            data = body.get('features')
        else:
            data = event.get('features')

        # 2. THE FIX: Flatten the list and strip any stray brackets
        # We want to ensure 'features' is just a simple list of numbers
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                # This handles [[7.4, 0.7...]] -> [7.4, 0.7...]
                features = data[0]
            else:
                # This handles [7.4, 0.7...]
                features = data
        else:
            return {"statusCode": 400, "body": json.dumps({"error": "Invalid features format"})}

        # 3. Convert to CSV string (Ensuring it is just 7.4,0.7,0.0...)
        csv_payload = ",".join(str(x) for x in features)
        
        # This will show up in your CloudWatch logs so you can see exactly what is sent
        print(f"Final Payload being sent to SageMaker: {csv_payload}")

        # 4. Call SageMaker
        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=csv_payload
        )
        
        result = response['Body'].read().decode('utf-8')
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'predicted_quality': round(float(result), 2)})
        }

    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
