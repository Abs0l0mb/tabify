'use strict';

import {
    PostgresFetcher,
    PostgresQueryBinder
} from '@src/classes';

export class TasksFetcher extends PostgresFetcher {

    constructor(private teamId: number) {

        super({
            columns: [
                { name: 'id', type: PostgresFetcher.NUMBER },
                { name: 'title', type: PostgresFetcher.STRING },
                { name: 'description', type: PostgresFetcher.STRING },
                { name: 'estimated_hours', type: PostgresFetcher.NUMBER },
                { name: 'category_title', type: PostgresFetcher.STRING },
                { name: 'creator', type: PostgresFetcher.STRING },
                { name: 'executor', type: PostgresFetcher.STRING },
                { name: 'status', type: PostgresFetcher.STRING },
                { name: 'time', type: PostgresFetcher.DATE }
            ]
        });
    }

    /*
    **
    **
    */
    protected getBaseQuery(binder: PostgresQueryBinder) : string {

        return `
            SELECT
                Task.id,
                Task.title,
                Task.description,
                Task.estimated_hours,
                Category.title AS category_title,
                Creator.last_name || ' ' || Creator.first_name AS creator,
                Executor.last_name || ' ' || Executor.first_name AS executor,
                LastEvent.status,
                LastEvent.time
            FROM tasks Task
            JOIN categories Category ON Category.id = Task.category_id
            JOIN accounts Creator ON Creator.id = Task.creator_account_id
            LEFT JOIN accounts Executor ON Executor.id = Task.executor_account_id
            LEFT JOIN LATERAL (
                SELECT status, time
                FROM task_events
                WHERE task_id = Task.id
                ORDER BY time DESC
                LIMIT 1
            ) LastEvent ON TRUE
            WHERE Task.team_id = ${binder.addParam(this.teamId)} AND LastEvent.status != 'ARCHIVED'
        `;
    }
}