'use strict';

import { BackendHttpServer } from '@src/classes';

declare const __MODE__: string;

process.env.MODE = __MODE__;
process.env.LOG_ID = 'main';
process.env.TIME_ZONE = 'Europe/Paris';

new BackendHttpServer(parseInt(process.argv[2]));