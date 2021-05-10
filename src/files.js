import { CognitoIdentityClient } from "@aws-sdk/client-cognito-identity";
import { fromCognitoIdentityPool } from "@aws-sdk/credential-provider-cognito-identity";
import { S3Client, ListObjectsCommand } from "@aws-sdk/client-s3";

const S3_BUCKET = "fogofwar";
const AWS_REGION = "us-east-2";
const TOPIC = "us_mainstream";
const LOCAL = false;

async function checkFileExists(key) {
  const client = new S3Client({
    region: AWS_REGION,
    credentials: fromCognitoIdentityPool({
      client: new CognitoIdentityClient({ region: AWS_REGION }),
      identityPoolId: "us-east-2:f4537342-bcf0-45e1-b055-b95ee3752b71", // IDENTITY_POOL_ID e.g., eu-west-1:xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxx
    }),
  });
  const data = await client.send(
    new ListObjectsCommand({
      Prefix: key,
      Bucket: S3_BUCKET,
    })
  );
  console.log(data);
}

let getS3FileName = function (name) {
  return `https://s3.${AWS_REGION}.amazonaws.com/${S3_BUCKET}/${name}`;
};

export function getName(date, interval, suffix) {
  let start = new Date(date);
  start.setUTCHours(start.getUTCHours() + 4);
  let end = new Date(start);
  end.setDate(start.getDate() + interval);
  let startStr = start.toISOString().slice(0, 10);
  let endStr = end.toISOString().slice(0, 10);
  let str = `${TOPIC}_${startStr}_${endStr}.${suffix}`;
  if (LOCAL) {
    return "/data/" + str;
  }
  return getS3FileName(str);
}

export function getEndDate(start, interval) {
  let end = new Date(start);
  end.setDate(end.getDate() + interval);
  return end.toISOString().slice(0, 10);
}
