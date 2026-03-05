'use strict';

import { SubProcess } from '@src/classes';

declare const __MODE__: string;

process.env.MODE = __MODE__;

process.env.LOG_ID = 'sub-process';
process.env.TIME_ZONE = 'Europe/Paris';

new SubProcess();