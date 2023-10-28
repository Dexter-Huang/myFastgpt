import type { FeConfigsType, SystemEnvType } from '@/types';
import type { NextApiRequest, NextApiResponse } from 'next';
import { jsonRes } from '@/service/response';
import { readFileSync } from 'fs';
import {
  type QAModelItemType,
  type ChatModelItemType,
  type VectorModelItemType,
  FunctionModelItemType
} from '@/types/model';

export type InitDateResponse = {
  chatModels: ChatModelItemType[];
  qaModel: QAModelItemType;
  vectorModels: VectorModelItemType[];
  feConfigs: FeConfigsType;
  systemVersion: string;
};

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (!global.feConfigs) {
    await getInitConfig();
  }
  jsonRes<InitDateResponse>(res, {
    data: {
      feConfigs: global.feConfigs,
      chatModels: global.chatModels,
      qaModel: global.qaModel,
      vectorModels: global.vectorModels,
      systemVersion: global.systemVersion || '0.0.0'
    }
  });
}

const defaultSystemEnv: SystemEnvType = {
  vectorMaxProcess: 15,
  qaMaxProcess: 15,
  pgIvfflatProbe: 20
};
const defaultFeConfigs: FeConfigsType = {
  show_emptyChat: true,
  show_contact: true,
  show_git: true,
  show_doc: true,
  show_register: true,
  systemTitle: 'SUSTech-HPC-Chat',
  authorText: 'source from FastGPT',
  limit: {
    exportLimitMinutes: 0
  },
  scripts: []
};
const defaultChatModels = [
  {
    model: 'baichuan2',
    name: 'baichuan2',
    contextMaxToken: 8000,
    quoteMaxToken: 4000,
    maxTemperature: 1.2,
    price: 0,
    defaultSystem: ''
  },
  {
    model: 'chatglm2',
    name: 'chatglm2',
    contextMaxToken: 8000,
    quoteMaxToken: 4000,
    maxTemperature: 1.2,
    price: 0,
    defaultSystem: ''
  },
  {
    model: 'chatglm3',
    name: 'chatglm3',
    contextMaxToken: 8000,
    quoteMaxToken: 4000,
    maxTemperature: 1.2,
    price: 0,
    defaultSystem: ''
  }
];
const defaultQAModel = {
  model: 'gpt-3.5-turbo-16k',
  name: 'GPT35-16k',
  maxToken: 16000,
  price: 0
};
export const defaultExtractModel: FunctionModelItemType = {
  model: 'gpt-3.5-turbo-16k',
  name: 'GPT35-16k',
  maxToken: 16000,
  price: 0,
  prompt: '',
  functionCall: true
};
export const defaultCQModel: FunctionModelItemType = {
  model: 'gpt-3.5-turbo-16k',
  name: 'GPT35-16k',
  maxToken: 16000,
  price: 0,
  prompt: '',
  functionCall: true
};
export const defaultQGModel: FunctionModelItemType = {
  model: 'gpt-3.5-turbo',
  name: 'FastAI-4k',
  maxToken: 4000,
  price: 1.5,
  prompt: '',
  functionCall: false
};

const defaultVectorModels: VectorModelItemType[] = [
  {
    model: 'text-embedding-ada-002',
    name: 'Embedding-2',
    price: 0,
    defaultToken: 500,
    maxToken: 3000
  }
];

export async function getInitConfig() {
  try {
    if (global.feConfigs) return;

    getSystemVersion();

    const filename =
      process.env.NODE_ENV === 'development' ? 'data/config.local.json' : '/app/data/config.json';
    const res = JSON.parse(readFileSync(filename, 'utf-8'));

    console.log(`System Version: ${global.systemVersion}`);

    console.log(res);

    global.systemEnv = res.SystemParams
      ? { ...defaultSystemEnv, ...res.SystemParams }
      : defaultSystemEnv;
    global.feConfigs = res.FeConfig ? { ...defaultFeConfigs, ...res.FeConfig } : defaultFeConfigs;
    global.chatModels = res.ChatModels || defaultChatModels;
    global.qaModel = res.QAModel || defaultQAModel;
    global.extractModel = res.ExtractModel || defaultExtractModel;
    global.cqModel = res.CQModel || defaultCQModel;
    global.qgModel = res.QGModel || defaultQGModel;
    global.vectorModels = res.VectorModels || defaultVectorModels;
  } catch (error) {
    setDefaultData();
    console.log('get init config error, set default', error);
  }
}

export function setDefaultData() {
  global.systemEnv = defaultSystemEnv;
  global.feConfigs = defaultFeConfigs;
  global.chatModels = defaultChatModels;
  global.qaModel = defaultQAModel;
  global.vectorModels = defaultVectorModels;
  global.extractModel = defaultExtractModel;
  global.cqModel = defaultCQModel;
  global.qgModel = defaultQGModel;
}

export function getSystemVersion() {
  try {
    if (process.env.NODE_ENV === 'development') {
      global.systemVersion = process.env.npm_package_version || '0.0.0';
      return;
    }
    const packageJson = JSON.parse(readFileSync('/app/package.json', 'utf-8'));

    global.systemVersion = packageJson?.version;
  } catch (error) {
    console.log(error);

    global.systemVersion = '0.0.0';
  }
}
