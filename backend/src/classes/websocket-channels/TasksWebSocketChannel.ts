'use strict';

import {
    WebSocketChannel,
    WebSocketInputRequest,
    WebSocketMessage,
    Parser,
    PublicError
} from '@src/classes';

export class TasksWebSocketChannel extends WebSocketChannel {

    private readonly teamChannels: Map<number, WebSocketChannel> = new Map();

    constructor() {

        super('tasks', true);

    }

    
}