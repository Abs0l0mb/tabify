'use strict';

process.env.LOG_ID = 'production-server';
process.env.START_TIME = Date.now();
process.env.TIME_ZONE = 'Europe/Paris';

import * as nodePath from 'path';
import * as nodeUrl from 'url';

import { HttpStaticServer } from './classes/http/HttpStaticServer.js';

const port = process.argv[2];

const dirname = nodePath.dirname(nodeUrl.fileURLToPath(import.meta.url));
const distDir = nodePath.resolve(dirname, '../dist/build');

const server = new HttpStaticServer();

server.static(distDir);
server.listen(port, '0.0.0.0');