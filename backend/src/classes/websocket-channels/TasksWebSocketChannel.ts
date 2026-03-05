'use strict';

import {
    WebSocketChannel,
    WebSocketInputRequest,
    WebSocketMessage,
    Parser,
    PublicError,
    Team,
    Task
} from '@src/classes';

export class TasksWebSocketChannel extends WebSocketChannel {

    private readonly teamChannels: Map<number, WebSocketChannel> = new Map();

    constructor() {

        super('tasks', true);

        this.request('/team/join', this.onJoinTeam.bind(this));
        this.request('/team/leave', this.onLeaveTeam.bind(this));
    }

    /*
    **
    **
    */
    public broadcastTaskSyncToTeam(team: Team, task: Task): void {
        
        const message: WebSocketMessage = {
            topic: '/task/sync',
            data: {
                taskId: task.id
            }
        };
        
        this.teamChannels.get(team.id!)?.broadcast(message);
    }

    /*
    **
    **
    */
    private async onJoinTeam(request: WebSocketInputRequest) : Promise<void> {

        const params = await Parser.parse(request.getData(), {
            teamId: Parser.integer
        });

        const team = new Team(params.teamId);
        
        await team.load();

        if (!team.data)
            throw new PublicError('team-not-found');
        
        if (!(await request.getClient().getRequest().account?.canAccessTeam(team)))
            throw new PublicError('not-allowed');

        if (!this.teamChannels.get(params.teamId))
            this.teamChannels.set(params.teamId, new WebSocketChannel(`team-${params.teamId}`));

        this.teamChannels.get(params.teamId)!.addClient(request.getClient());

        request.respondSuccessContent();
    }

    /*
    **
    **
    */
    private async onLeaveTeam(request: WebSocketInputRequest) : Promise<void> {

        const params = await Parser.parse(request.getData(), {
            teamId: Parser.integer
        });

        this.teamChannels.get(params.teamId)?.removeClient(request.getClient());
            
        request.respondSuccessContent();
    }
}