import os 
import grpc
from google.protobuf.json_format import MessageToDict
from pydub import AudioSegment

from speaker_verification import speaker_verification_pb2
from speaker_verification import speaker_verification_pb2_grpc

SV_HOST =  os.environ.get('SV_HOST', 'localhost')
SV_PORT = os.environ.get('SV_PORT', '5001')

CHANNEL_IP = f"{SV_HOST}:{SV_PORT}"

def verify(enroll_utts, candidate_utt):
    channel = grpc.insecure_channel(CHANNEL_IP)
    stub = speaker_verification_pb2_grpc.SpeakerVerificationServiceStub(channel)
    request = speaker_verification_pb2.SpeakerVerificationRequest(
        enroll_utterances=enroll_utts, utterance=candidate_utt)
    ret = stub.SpeakerVerify(request)
    print(ret)

def test():
    import os

    enroll_dir = 'sample_test/si/enroll'
    candidate_dir= 'sample_test/si/pred/'
    enroll_utts = []
    for root, dirs, files in os.walk(enroll_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            with open(filepath, 'rb') as f:
                enroll_utts.append(f.read())

    for root, dirs, files in os.walk(candidate_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            with open(filepath, 'rb') as f:
                candidate_utt = f.read()
                print('Candidate: ',filepath)
                verify(enroll_utts, candidate_utt)
                

if __name__ == "__main__":
    test()


